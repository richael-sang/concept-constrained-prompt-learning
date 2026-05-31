import json
import os.path as osp
import random
import re

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "CoOp",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def normalize_name(name):
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = name.replace("_", " ").replace("-", " ").replace("/", " ")
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CCPL.N_CTX
        ctx_init = cfg.TRAINER.CCPL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.CCPL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.CCPL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Invalid CLASS_TOKEN_POSITION: {self.class_token_position}")

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.prompt_learner = PromptLearner(cfg, self.classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        concept_prompt_lists, avg_concepts, fallback_count = self._build_concept_prompt_lists(
            cfg, self.classnames
        )
        self.concept_prompt_lists = concept_prompt_lists
        self.avg_concepts = avg_concepts
        self.fallback_count = fallback_count
        self.max_concepts = max(len(x) for x in concept_prompt_lists)
        self._concept_bank_built = False

        print(
            "[CCPL] Concept bank loaded: dataset={}, classes={}, avg_concepts={:.2f}, "
            "fallback_classes={}, max_concepts={}".format(
                cfg.DATASET.NAME,
                len(self.classnames),
                self.avg_concepts,
                self.fallback_count,
                self.max_concepts,
            )
        )

    def _resolve_concept_bank_path(self, cfg):
        raw_path = cfg.TRAINER.CCPL.CONCEPT_BANK
        if osp.isabs(raw_path):
            return raw_path
        repo_root = osp.dirname(osp.dirname(__file__))
        return osp.join(repo_root, raw_path)

    def _read_concept_bank(self, cfg):
        bank_path = self._resolve_concept_bank_path(cfg)
        if not osp.exists(bank_path):
            raise FileNotFoundError(f"CCPL concept bank not found: {bank_path}")
        with open(bank_path, "r", encoding="utf-8") as f:
            return json.load(f), bank_path

    def _match_dataset_key(self, bank, dataset_name):
        normalized_map = {normalize_name(k): k for k in bank.keys()}
        candidates = [
            dataset_name,
            normalize_name(dataset_name),
            "DescribableTextures",
            "DTD",
            "dtd",
        ]
        for item in candidates:
            nk = normalize_name(item)
            if nk in normalized_map:
                return normalized_map[nk]
        return None

    def _build_fallback_concepts(self, class_name, template_list):
        cname = class_name.replace("_", " ")
        return [t.format(class_name=cname) for t in template_list]

    def _lookup_concepts(self, normalized_class_name, norm_concept_map):
        concept_list = norm_concept_map.get(normalized_class_name, None)
        if concept_list is not None and len(concept_list) > 0:
            return concept_list

        class_tokens = set(normalized_class_name.split())
        best_key = None
        best_score = -1
        for candidate_key, candidate_concepts in norm_concept_map.items():
            if candidate_concepts is None or len(candidate_concepts) == 0:
                continue
            cand_tokens = set(candidate_key.split())
            overlap = len(class_tokens & cand_tokens)
            # Prefer candidate concepts whose tokens are mostly covered by class name.
            if overlap == 0:
                continue
            covered = overlap == len(cand_tokens) or overlap == len(class_tokens)
            if not covered:
                continue
            score = overlap
            if score > best_score:
                best_score = score
                best_key = candidate_key

        if best_key is not None:
            return norm_concept_map[best_key]
        return None

    def _build_concept_prompt_lists(self, cfg, classnames):
        bank, bank_path = self._read_concept_bank(cfg)
        dataset_name = cfg.DATASET.NAME
        dataset_key = self._match_dataset_key(bank, dataset_name)
        if dataset_key is None:
            raise KeyError(
                f"Dataset {dataset_name} not found in concept bank: {bank_path}. "
                "Please add a matching entry."
            )

        dataset_bank = bank[dataset_key]
        norm_concept_map = {normalize_name(k): v for k, v in dataset_bank.items() if k != "__template__"}
        template_list = dataset_bank.get(
            "__template__",
            [
                "{class_name} texture",
                "{class_name} surface pattern",
                "{class_name} visual texture",
                "{class_name} material pattern",
                "{class_name} repeated texture",
            ],
        )

        all_prompts = []
        fallback_count = 0
        for cname in classnames:
            key = normalize_name(cname)
            concept_list = self._lookup_concepts(key, norm_concept_map)
            if concept_list is None or len(concept_list) == 0:
                concept_list = self._build_fallback_concepts(cname, template_list)
                fallback_count += 1

            class_prompts = [
                f"a photo of a {cname}, which contains {concept}." for concept in concept_list
            ]
            all_prompts.append(class_prompts)

        avg_concepts = sum(len(x) for x in all_prompts) / float(len(all_prompts))
        return all_prompts, avg_concepts, fallback_count

    def _build_concept_feature_bank(self, device):
        n_cls = len(self.concept_prompt_lists)
        token_bank = torch.zeros(n_cls, self.max_concepts, 77, dtype=torch.long, device=device)
        concept_mask = torch.zeros(n_cls, self.max_concepts, dtype=torch.bool, device=device)

        all_tokens = []
        all_indices = []
        for class_idx, prompts in enumerate(self.concept_prompt_lists):
            for concept_idx, prompt in enumerate(prompts):
                token_bank[class_idx, concept_idx] = clip.tokenize(prompt).squeeze(0).to(device)
                concept_mask[class_idx, concept_idx] = True
                all_tokens.append(token_bank[class_idx, concept_idx].unsqueeze(0))
                all_indices.append((class_idx, concept_idx))

        all_tokens = torch.cat(all_tokens, dim=0)
        with torch.no_grad():
            concept_features = self.clip_model.encode_text(all_tokens)
            concept_features = concept_features / concept_features.norm(dim=-1, keepdim=True)
            concept_features = concept_features.type(self.dtype)

        feature_bank = torch.zeros(
            n_cls, self.max_concepts, concept_features.shape[-1], dtype=self.dtype, device=device
        )
        for feat, (class_idx, concept_idx) in zip(concept_features, all_indices):
            feature_bank[class_idx, concept_idx] = feat

        self.register_buffer("concept_mask", concept_mask, persistent=False)
        self.register_buffer("concept_feature_bank", feature_bank, persistent=False)
        self._concept_bank_built = True

    def _get_concept_text_features(self):
        if not self._concept_bank_built:
            self._build_concept_feature_bank(self.logit_scale.device)

        n_cls = self.concept_feature_bank.shape[0]
        dropout = self.cfg.TRAINER.CCPL.CONCEPT_DROPOUT
        is_training = self.training and dropout > 0
        concept_features = []
        for class_idx in range(n_cls):
            valid_ids = torch.where(self.concept_mask[class_idx])[0]
            if is_training and len(valid_ids) > 1:
                keep_flags = torch.rand(len(valid_ids), device=valid_ids.device) > dropout
                if keep_flags.sum() == 0:
                    rand_id = random.randint(0, len(valid_ids) - 1)
                    keep_flags[rand_id] = True
                selected = valid_ids[keep_flags]
            else:
                selected = valid_ids

            class_feats = self.concept_feature_bank[class_idx, selected]
            class_feat = class_feats.mean(dim=0)
            class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)
            concept_features.append(class_feat)

        concept_features = torch.stack(concept_features, dim=0)
        return concept_features

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        class_text_features = self.text_encoder(prompts, self.tokenized_prompts)
        class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        concept_text_features = self._get_concept_text_features().detach()

        logit_scale = self.logit_scale.exp()
        class_logits = logit_scale * image_features @ class_text_features.t()
        concept_logits = logit_scale * image_features @ concept_text_features.t()
        alpha = self.cfg.TRAINER.CCPL.ENSEMBLE_ALPHA
        final_logits = (1.0 - alpha) * class_logits + alpha * concept_logits

        if self.prompt_learner.training:
            if label is None:
                raise ValueError("Label is required during training for CCPL")
            loss_ce = F.cross_entropy(class_logits, label)
            cosine_sim = F.cosine_similarity(class_text_features, concept_text_features, dim=-1)
            loss_text = (1.0 - cosine_sim).mean()
            kl_t = self.cfg.TRAINER.CCPL.KL_T
            loss_kl = F.kl_div(
                F.log_softmax(class_logits / kl_t, dim=-1),
                F.softmax(concept_logits / kl_t, dim=-1),
                reduction="batchmean",
            ) * (kl_t ** 2)

            loss = (
                loss_ce
                + self.cfg.TRAINER.CCPL.LAMBDA_TEXT * loss_text
                + self.cfg.TRAINER.CCPL.LAMBDA_KL * loss_kl
            )
            return {
                "loss": loss,
                "loss_ce": loss_ce,
                "loss_text": loss_text,
                "loss_kl": loss_kl,
                "class_logits": class_logits,
                "final_logits": final_logits,
            }

        return final_logits


@TRAINER_REGISTRY.register()
class CCPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CCPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.CCPL.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CCPL model")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CCPL.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.CCPL.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = output["loss"]
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            final_logits = output["final_logits"]
            loss_ce = output["loss_ce"]
            loss_text = output["loss_text"]
            loss_kl = output["loss_kl"]
        else:
            output = self.model(image, label)
            loss = output["loss"]
            self.model_backward_and_update(loss)
            final_logits = output["final_logits"]
            loss_ce = output["loss_ce"]
            loss_text = output["loss_text"]
            loss_kl = output["loss_kl"]

        loss_summary = {
            "loss_total": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_text": loss_text.item(),
            "loss_kl": loss_kl.item(),
            "acc": compute_accuracy(final_logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input_tensor = batch["img"]
        label = batch["label"]
        input_tensor = input_tensor.to(self.device)
        label = label.to(self.device)
        return input_tensor, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            loaded_epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

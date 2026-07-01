"""NAF-specific model components for multi-level classification."""

import torch

from torchTextClassifiers.model.components import (
    CategoricalForwardType,
    ClassificationHead,
    SentenceEmbedder,
    TokenEmbedder,
)
from torchTextClassifiers.model.components.text_embedder import (
    LabelAttentionConfig,
    SentenceEmbedderConfig,
    TokenEmbedderConfig,
)

from src.multilevel import MultiLevelTextClassificationModel


class _ZeroCatNetForward(torch.nn.Module):
    """No-op categorical net — supports categorical vars but none are used for NAF."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.categorical_vocabulary_sizes = []
        self.forward_type = CategoricalForwardType.SUM_TO_TEXT
        self.output_dim = emb_dim
        self._emb_dim = emb_dim

    def forward(self, x=None):
        return None


class NAFMultiLevelModel(MultiLevelTextClassificationModel):
    """MultiLevelTextClassificationModel patched for zero categorical inputs."""

    def forward(self, input_ids, attention_mask, categorical_vars=None, **kwargs):
        token_out = self.token_embedder(input_ids, attention_mask)
        x_token = token_out["token_embeddings"]

        outputs = []
        for sentence_embedder, classification_head in zip(
            self.sentence_embedders, self.classification_heads
        ):
            sent_out = sentence_embedder(
                token_embeddings=x_token, attention_mask=attention_mask
            )
            x_combined = sent_out["sentence_embedding"]
            outputs.append(classification_head(x_combined).squeeze(-1))

        return outputs


def build_model(
    tokenizer,
    num_classes_per_level: list[int],
    emb_dim: int,
    n_heads_label_attention: int | None,
) -> NAFMultiLevelModel:
    """Instantiate a NAFMultiLevelModel from config parameters."""
    token_config = TokenEmbedderConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=emb_dim,
        padding_idx=tokenizer.padding_idx,
        attention_config=None,
    )
    token_embedder = TokenEmbedder(token_config)

    sentence_embedders = []
    classification_heads = []

    for n_classes in num_classes_per_level:
        if n_heads_label_attention:
            la_config = LabelAttentionConfig(
                n_head=n_heads_label_attention,
                num_classes=n_classes,
                embedding_dim=emb_dim,
            )
            sent_cfg = SentenceEmbedderConfig(
                aggregation_method=None,
                label_attention_config=la_config,
            )
            head = ClassificationHead(input_dim=emb_dim, num_classes=1)
        else:
            sent_cfg = SentenceEmbedderConfig(aggregation_method="mean")
            head = ClassificationHead(input_dim=emb_dim, num_classes=n_classes)

        sentence_embedders.append(SentenceEmbedder(sent_cfg))
        classification_heads.append(head)

    return NAFMultiLevelModel(
        token_embedder=token_embedder,
        sentence_embedders=sentence_embedders,
        classification_heads=classification_heads,
        categorical_variable_net=_ZeroCatNetForward(emb_dim),
    )

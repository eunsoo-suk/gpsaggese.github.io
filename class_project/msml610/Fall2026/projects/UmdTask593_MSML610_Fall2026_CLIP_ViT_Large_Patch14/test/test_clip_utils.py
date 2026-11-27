"""
Import as:

import test.test_clip_utils as ttcliut
"""

import logging

import numpy as np
import torch

import helpers.hunit_test as hunitest

import clip_utils as cliputil

_LOG = logging.getLogger(__name__)

# Using tiny sizes to make the tests run fast.
_BATCH = 4
_N_CLASSES = 3


# #############################################################################
# Test_majority_vote
# #############################################################################


class Test_majority_vote(hunitest.TestCase):
    """
    Test `clip_utils._majority_vote()`.
    """

    def test_two_agree(self) -> None:
        """
        2 of 3 annotators agree, so their label wins.
        """
        # Preparing the inputs.
        votes = ["positive", "neutral", "positive"]
        # Running test.
        actual = cliputil._majority_vote(votes)
        # Checking outputs.
        self.assertEqual(actual, "positive")

    def test_no_majority(self) -> None:
        """
        All 3 annotators disagree, so there is no label.
        """
        # Preparing inputs.

        votes = ["positive", "neutral", "negative"]

        # Running test.

        actual = cliputil._majority_vote(votes)

        # Checking outputs.

        self.assertIsNone(actual)


# #############################################################################
# Test_combine_text_image
# #############################################################################


class Test_combine_text_image(hunitest.TestCase):
    """
    Test `clip_utils._combine_text_image()`.
    """

    def test_same_label(self) -> None:
        """
        Same label on text and image is kept.
        """
        actual = cliputil._combine_text_image("negative", "negative")
        self.assertEqual(actual, "negative")

    def test_neutral_text(self) -> None:
        """
        Neutral text takes the image label.
        """
        actual = cliputil._combine_text_image("neutral", "positive")
        self.assertEqual(actual, "positive")

    def test_neutral_image(self) -> None:
        """
        Neutral image takes the text label.
        """
        actual = cliputil._combine_text_image("negative", "neutral")
        self.assertEqual(actual, "negative")

    def test_conflict_dropped(self) -> None:
        """
        Positive vs negative is a conflict, so the post is dropped.
        """
        actual = cliputil._combine_text_image("positive", "negative")
        self.assertIsNone(actual)

    def test_missing_side_dropped(self) -> None:
        """
        If one side has no label, the post is dropped.
        """
        self.assertIsNone(cliputil._combine_text_image(None, "positive"))
        self.assertIsNone(cliputil._combine_text_image("positive", None))


# #############################################################################
# Test_head_output_shapes
# #############################################################################


class Test_head_output_shapes(hunitest.TestCase):
    """
    Each head must output 3 scores per post: shape (batch, 3).
    """

    @staticmethod
    def _set_seed() -> None:
        """
        Set seeds so every run gives the same numbers.
        """
        torch.manual_seed(0)
        np.random.seed(0)

    def test_linear_head(self) -> None:
        """
        Linear baseline from `make_linear_head()`.
        """
        # Preparing inputs.

        self._set_seed()
        dim = 16
        n_posts, n_train = 10, 8
        X = np.random.randn(n_posts, dim).astype(np.float32)
        train_idx = np.arange(n_train)
        head = cliputil.make_linear_head(X, train_idx).eval()
        x = torch.randn(_BATCH, dim)

        # Running test.

        logits = head(x)

        # Checking outputs.

        self.assertEqual(tuple(logits.shape), (_BATCH, _N_CLASSES))

    def test_stack_head(self) -> None:
        """
        H1 stack head from `make_stack_head()`.
        """
        # Preparing inputs.

        self._set_seed()
        dim = 16
        n_posts, n_train = 10, 8
        emb = {
            "image": np.random.randn(n_posts, dim).astype(np.float32),
            "text": np.random.randn(n_posts, dim).astype(np.float32),
        }
        train_idx = np.arange(n_train)
        head = cliputil.make_stack_head(emb, train_idx).eval()
        txt = torch.randn(_BATCH, dim)
        img = torch.randn(_BATCH, dim)

        # Running test.
        logits = head(txt, img)

        # Checking the outputs.
        self.assertEqual(tuple(logits.shape), (_BATCH, _N_CLASSES))

    def test_gated_fusion_head(self) -> None:
        """
        H2 gated head: output shape, and the gate stays in [0.05, 0.95].
        """
        # Preparing the inputs.

        self._set_seed()
        dim = 16
        head = cliputil.GatedFusionHead(
            in_dim=dim, hidden=8, ffn_dim=16, cls_dim=8
        ).eval()

        # The gate starts at 0.5, so using the big random weights to push it to
        # the limits.

        torch.nn.init.normal_(head.gate.weight, std=10.0)
        torch.nn.init.normal_(head.gate.bias, std=10.0)

        # CLIP embeddings are normalized, so normalizing the fake ones too.

        txt = torch.nn.functional.normalize(torch.randn(_BATCH, dim), dim=-1)
        img = torch.nn.functional.normalize(torch.randn(_BATCH, dim), dim=-1)

        # Running test.
        logits = head(txt, img)

        # Checking the outputs.
        self.assertEqual(tuple(logits.shape), (_BATCH, _N_CLASSES))
        gate = head.last_gate
        self.assertEqual(tuple(gate.shape), (_BATCH, 1))

        # Small tolerance for float rounding.

        self.assertGreaterEqual(gate.min().item(), 0.05 - 1e-6)
        self.assertLessEqual(gate.max().item(), 0.95 + 1e-6)

    def test_cross_attention_head(self) -> None:
        """
        H3 cross-attention head with tweets of different lengths.
        """

        # Preparing inputs.

        self._set_seed()
        img_dim, txt_dim = 32, 24
        n_img_tok, n_txt_tok = 5, 7
        head = cliputil.TokenCrossAttentionHead(
            img_dim=img_dim, txt_dim=txt_dim, d=16, heads=2, ffn_dim=32
        ).eval()
        img_tok = torch.randn(_BATCH, n_img_tok, img_dim)
        txt_tok = torch.randn(_BATCH, n_txt_tok, txt_dim)

        # Different tweet lengths: 7, 5, 3 and 1 real tokens.

        lengths = torch.tensor([7, 5, 3, 1])
        txt_mask = (torch.arange(n_txt_tok)[None, :] < lengths[:, None]).long()

        # Running test.

        logits = head(img_tok, txt_tok, txt_mask)

        # Checking outputs.

        self.assertEqual(tuple(logits.shape), (_BATCH, _N_CLASSES))
        self.assertTrue(torch.isfinite(logits).all().item())


# #############################################################################
# Test_evaluate
# #############################################################################


class Test_evaluate(hunitest.TestCase):
    """
    Test `clip_utils.evaluate()`.
    """

    def test_perfect_predictions(self) -> None:
        """
        Perfect predictions give accuracy and macro F1 of 1.0.
        """

        # Preparing inputs.

        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = y_true.copy()
        # Run test.
        metrics = cliputil.evaluate(y_true, y_pred)

        # Checking outputs.

        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
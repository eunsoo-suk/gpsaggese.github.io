"""
Import as:

import test.test_clip_utils as ttcliut
"""

import logging
import pprint
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import helpers.hdbg as hdbg
import helpers.hunit_test as hunitest

import clip_utils as cliputil

_LOG = logging.getLogger(__name__)


# #############################################################################
# Helpers
# #############################################################################


# Use tiny sizes so the tests run fast.
_DIM = 16
_N_POSTS = 10
_N_TRAIN = 8
_N_CLASSES = 3
# Number of digits used when comparing floats.
_N_DIGITS = 4


def _set_seed() -> None:
    """
    Set seeds so every run gives the same numbers.
    """
    torch.manual_seed(0)
    np.random.seed(0)


# #############################################################################
# Test_majority_vote
# #############################################################################


class Test_majority_vote(hunitest.TestCase):
    """
    Test `clip_utils._majority_vote()`.
    """

    def helper(self, votes: List[str], expected: Optional[str]) -> None:
        """
        Run `_majority_vote()` and compare with the expected label.

        :param votes: labels from the 3 annotators
        :param expected: expected majority label, or `None`
        """
        # Run test.
        actual = cliputil._majority_vote(votes)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        2 of 3 annotators agree, so their label wins.
        """
        # Prepare inputs.
        votes = ["positive", "neutral", "positive"]
        # Prepare outputs.
        expected = "positive"
        # Run test.
        self.helper(votes, expected)

    def test2(self) -> None:
        """
        All 3 annotators disagree, so there is no label.
        """
        # Prepare inputs.
        votes = ["positive", "neutral", "negative"]
        # Prepare outputs.
        expected = None
        # Run test.
        self.helper(votes, expected)

    def test3(self) -> None:
        """
        All 3 annotators agree, so their label is returned.
        """
        # Prepare inputs.
        votes = ["negative", "negative", "negative"]
        # Prepare outputs.
        expected = "negative"
        # Run test.
        self.helper(votes, expected)


# #############################################################################
# Test_combine_text_image
# #############################################################################


class Test_combine_text_image(hunitest.TestCase):
    """
    Test `clip_utils._combine_text_image()`.
    """

    def helper(
        self,
        text_label: Optional[str],
        image_label: Optional[str],
        expected: Optional[str],
    ) -> None:
        """
        Run `_combine_text_image()` and compare with the expected label.

        :param text_label: majority label of the text
        :param image_label: majority label of the image
        :param expected: expected post label, or `None` if dropped
        """
        # Run test.
        actual = cliputil._combine_text_image(text_label, image_label)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        Same label on text and image is kept.
        """
        # Prepare inputs.
        text_label = "negative"
        image_label = "negative"
        # Prepare outputs.
        expected = "negative"
        # Run test.
        self.helper(text_label, image_label, expected)

    def test2(self) -> None:
        """
        Neutral text takes the image label.
        """
        # Prepare inputs.
        text_label = "neutral"
        image_label = "positive"
        # Prepare outputs.
        expected = "positive"
        # Run test.
        self.helper(text_label, image_label, expected)

    def test3(self) -> None:
        """
        Neutral image takes the text label.
        """
        # Prepare inputs.
        text_label = "negative"
        image_label = "neutral"
        # Prepare outputs.
        expected = "negative"
        # Run test.
        self.helper(text_label, image_label, expected)

    def test4(self) -> None:
        """
        Positive vs negative is a conflict, so the post is dropped.
        """
        # Prepare inputs.
        text_label = "positive"
        image_label = "negative"
        # Prepare outputs.
        expected = None
        # Run test.
        self.helper(text_label, image_label, expected)

    def test5(self) -> None:
        """
        If the text has no label, the post is dropped.
        """
        # Prepare inputs.
        text_label = None
        image_label = "positive"
        # Prepare outputs.
        expected = None
        # Run test.
        self.helper(text_label, image_label, expected)

    def test6(self) -> None:
        """
        If the image has no label, the post is dropped.
        """
        # Prepare inputs.
        text_label = "positive"
        image_label = None
        # Prepare outputs.
        expected = None
        # Run test.
        self.helper(text_label, image_label, expected)


# #############################################################################
# Test_make_linear_head
# #############################################################################


class Test_make_linear_head(hunitest.TestCase):
    """
    Test `clip_utils.make_linear_head()`.
    """

    def helper(self, batch: int, expected: Tuple[int, int]) -> None:
        """
        Build the head on fake features and check the output shape.

        :param batch: number of posts to score
        :param expected: expected output shape
        """
        # Prepare inputs.
        _set_seed()
        X = np.random.randn(_N_POSTS, _DIM).astype(np.float32)
        train_idx = np.arange(_N_TRAIN)
        x = torch.randn(batch, _DIM)
        # Run test.
        head = cliputil.make_linear_head(X, train_idx).eval()
        logits = head(x)
        # Check outputs.
        self.assert_equal(str(tuple(logits.shape)), str(expected))

    def test1(self) -> None:
        """
        A batch of 4 posts gives 3 scores per post.
        """
        # Prepare inputs.
        batch = 4
        # Prepare outputs.
        expected = (batch, _N_CLASSES)
        # Run test.
        self.helper(batch, expected)

    def test2(self) -> None:
        """
        A single post still gives 3 scores.
        """
        # Prepare inputs.
        batch = 1
        # Prepare outputs.
        expected = (batch, _N_CLASSES)
        # Run test.
        self.helper(batch, expected)


# #############################################################################
# Test_make_stack_head
# #############################################################################


class Test_make_stack_head(hunitest.TestCase):
    """
    Test `clip_utils.make_stack_head()`.
    """

    def helper(self, batch: int, expected: Tuple[int, int]) -> None:
        """
        Build the H1 head on fake embeddings and check the output shape.

        :param batch: number of posts to score
        :param expected: expected output shape
        """
        # Prepare inputs.
        _set_seed()
        emb = {
            "image": np.random.randn(_N_POSTS, _DIM).astype(np.float32),
            "text": np.random.randn(_N_POSTS, _DIM).astype(np.float32),
        }
        train_idx = np.arange(_N_TRAIN)
        txt = torch.randn(batch, _DIM)
        img = torch.randn(batch, _DIM)
        # Run test.
        head = cliputil.make_stack_head(emb, train_idx).eval()
        logits = head(txt, img)
        # Check outputs.
        self.assert_equal(str(tuple(logits.shape)), str(expected))

    def test1(self) -> None:
        """
        A batch of 4 posts gives 3 scores per post.
        """
        # Prepare inputs.
        batch = 4
        # Prepare outputs.
        expected = (batch, _N_CLASSES)
        # Run test.
        self.helper(batch, expected)

    def test2(self) -> None:
        """
        A single post still gives 3 scores.
        """
        # Prepare inputs.
        batch = 1
        # Prepare outputs.
        expected = (batch, _N_CLASSES)
        # Run test.
        self.helper(batch, expected)


# #############################################################################
# TestGatedFusionHead
# #############################################################################


class TestGatedFusionHead(hunitest.TestCase):
    """
    Test `clip_utils.GatedFusionHead`.
    """

    def helper(
        self, batch: int, anchor: str, gate_std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the H2 head on fake normalized embeddings.

        :param batch: number of posts to score
        :param anchor: main modality of the head, "text" or "image"
        :param gate_std: std of the gate weights; 0.0 keeps the zero init
        :return: the logits and the gate values
        """
        # Prepare inputs.
        _set_seed()
        hidden = 8
        ffn_dim = 16
        cls_dim = 8
        norm_dim = -1
        head = cliputil.GatedFusionHead(
            anchor=anchor,
            in_dim=_DIM,
            hidden=hidden,
            ffn_dim=ffn_dim,
            cls_dim=cls_dim,
        ).eval()
        # The gate starts at 0.5, so use big random weights to push it to
        # the limits. A std of 0.0 gives back the zero init.
        torch.nn.init.normal_(head.gate.weight, std=gate_std)
        torch.nn.init.normal_(head.gate.bias, std=gate_std)
        # CLIP embeddings are normalized, so normalize the fake ones too.
        txt = torch.nn.functional.normalize(
            torch.randn(batch, _DIM), dim=norm_dim
        )
        img = torch.nn.functional.normalize(
            torch.randn(batch, _DIM), dim=norm_dim
        )
        # Run test.
        logits = head(txt, img)
        # The head stores the gate after each forward pass.
        gate = head.last_gate
        hdbg.dassert_is_not(gate, None)
        return logits, gate

    def test1(self) -> None:
        """
        A batch of 4 posts gives 3 scores per post.
        """
        # Prepare inputs.
        batch = 4
        anchor = "text"
        gate_std = 0.0
        # Prepare outputs.
        expected = (batch, _N_CLASSES)
        # Run test.
        logits, _ = self.helper(batch, anchor, gate_std)
        # Check outputs.
        self.assert_equal(str(tuple(logits.shape)), str(expected))

    def test2(self) -> None:
        """
        The zero-initialized gate starts at (0.05 + 0.95) / 2 = 0.5.
        """
        # Prepare inputs.
        batch = 4
        anchor = "text"
        gate_std = 0.0
        # Prepare outputs.
        expected = [0.5] * batch
        # Run test.
        _, gate = self.helper(batch, anchor, gate_std)
        # Check outputs.
        actual = [round(g, _N_DIGITS) for g in gate.squeeze(1).tolist()]
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        With big random gate weights, the gate still stays in [0.05, 0.95].
        """
        # Prepare inputs.
        batch = 4
        anchor = "text"
        gate_std = 10.0
        # Small tolerance for float rounding.
        tol = 1e-6
        gate_min = 0.05 - tol
        gate_max = 0.95 + tol
        # Prepare outputs.
        expected = True
        # Run test.
        _, gate = self.helper(batch, anchor, gate_std)
        # Check outputs.
        actual = bool(((gate >= gate_min) & (gate <= gate_max)).all())
        self.assert_equal(str(actual), str(expected))

    def test4(self) -> None:
        """
        With the image as the anchor, the head still gives 3 scores per post.
        """
        # Prepare inputs.
        batch = 4
        anchor = "image"
        gate_std = 0.0
        # Prepare outputs.
        expected = (batch, _N_CLASSES)
        # Run test.
        logits, _ = self.helper(batch, anchor, gate_std)
        # Check outputs.
        self.assert_equal(str(tuple(logits.shape)), str(expected))


# #############################################################################
# TestTokenCrossAttentionHead
# #############################################################################


class TestTokenCrossAttentionHead(hunitest.TestCase):
    """
    Test `clip_utils.TokenCrossAttentionHead`.
    """

    def helper(self, lengths: List[int]) -> torch.Tensor:
        """
        Run the H3 head on fake tokens, padding the text to the longest tweet.

        :param lengths: number of real text tokens for each post
        :return: the logits
        """
        # Prepare inputs.
        _set_seed()
        img_dim, txt_dim = 32, 24
        n_img_tok = 5
        n_txt_tok = max(lengths)
        batch = len(lengths)
        d = 16
        heads = 2
        ffn_dim = 32
        head = cliputil.TokenCrossAttentionHead(
            img_dim=img_dim, txt_dim=txt_dim, d=d, heads=heads, ffn_dim=ffn_dim
        ).eval()
        # Random image patches and text tokens for each post.
        img_tok = torch.randn(batch, n_img_tok, img_dim)
        txt_tok = torch.randn(batch, n_txt_tok, txt_dim)
        # Tweets have different lengths: pad the text to the longest tweet and
        # mark the real tokens with 1 and the padding with 0.
        lengths_t = torch.tensor(lengths)
        positions = torch.arange(n_txt_tok)
        txt_mask = (positions[None, :] < lengths_t[:, None]).long()
        # Run test.
        logits = head(img_tok, txt_tok, txt_mask)
        return logits

    def test1(self) -> None:
        """
        Tweets of different lengths give 3 scores per post.
        """
        # Prepare inputs.
        lengths = [7, 5, 3, 1]
        # Prepare outputs.
        expected = (len(lengths), _N_CLASSES)
        # Run test.
        logits = self.helper(lengths)
        # Check outputs.
        self.assert_equal(str(tuple(logits.shape)), str(expected))

    def test2(self) -> None:
        """
        A tweet with only 1 real token does not produce NaN or inf scores.
        """
        # Prepare inputs.
        lengths = [7, 1]
        # Prepare outputs.
        expected = True
        # Run test.
        logits = self.helper(lengths)
        # Check outputs.
        actual = bool(torch.isfinite(logits).all())
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        A single post gives 3 scores.
        """
        # Prepare inputs.
        lengths = [4]
        # Prepare outputs.
        expected = (len(lengths), _N_CLASSES)
        # Run test.
        logits = self.helper(lengths)
        # Check outputs.
        self.assert_equal(str(tuple(logits.shape)), str(expected))


# #############################################################################
# Test_evaluate
# #############################################################################


class Test_evaluate(hunitest.TestCase):
    """
    Test `clip_utils.evaluate()`.
    """

    def helper(
        self, y_true: np.ndarray, y_pred: np.ndarray, expected: str
    ) -> None:
        """
        Run `evaluate()` and compare all the metrics at once.

        :param y_true: true class ids
        :param y_pred: predicted class ids
        :param expected: expected metrics, formatted with `pprint`
        """
        # Run test.
        metrics = cliputil.evaluate(y_true, y_pred)
        # Round so the comparison does not depend on float noise.
        rounded: Dict[str, float] = {
            k: round(float(v), _N_DIGITS) for k, v in metrics.items()
        }
        actual = pprint.pformat(rounded)
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Perfect predictions give 1.0 for every metric.
        """
        # Prepare inputs.
        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = y_true.copy()
        # Prepare outputs.
        expected = """
        {'accuracy': 1.0,
         'macro_f1': 1.0,
         'precision_negative': 1.0,
         'precision_neutral': 1.0,
         'precision_positive': 1.0,
         'recall_negative': 1.0,
         'recall_neutral': 1.0,
         'recall_positive': 1.0}
        """
        # Run test.
        self.helper(y_true, y_pred, expected)

    def test2(self) -> None:
        """
        Always predicting the majority class gives high accuracy but low macro
        F1.
        """
        # Prepare inputs.
        # 4 of 6 posts are positive (class 2), and we always predict positive.
        positive_class = 2
        y_true = np.array([0, 1, 2, 2, 2, 2])
        y_pred = np.full_like(y_true, positive_class)
        # Prepare outputs.
        # Positive F1 is 0.8, the other classes get 0, so macro F1 = 0.8 / 3.
        expected = """
        {'accuracy': 0.6667,
         'macro_f1': 0.2667,
         'precision_negative': 0.0,
         'precision_neutral': 0.0,
         'precision_positive': 0.6667,
         'recall_negative': 0.0,
         'recall_neutral': 0.0,
         'recall_positive': 1.0}
        """
        # Run test.
        self.helper(y_true, y_pred, expected)

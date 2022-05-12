from typing import List, Optional, Tuple

import torch
from torch import nn


class CRF(nn.Module):
    """Linear-chain Conditional Random Field (CRF).

    Args:
        num_classes: Number of valid classes for prediction.
        batch_first: Whether the first dimension represents the batch dimension.
        rules (optional): A list of transitions and their weights to use when initializing the transition scores.
    """

    FIXED_CLASSES = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
    }

    def __init__(
        self,
        num_classes: int,
        batch_first: bool = True,
        rules: Optional[List[Tuple[int, int, float]]] = None,
    ):
        super(CRF, self).__init__()
        self.PAD_TAG_IDX = self.FIXED_CLASSES["<PAD>"]
        self.BOS_TAG_IDX = self.FIXED_CLASSES["<BOS>"]
        self.EOS_TAG_IDX = self.FIXED_CLASSES["<EOS>"]
        self.num_classes = num_classes + len(self.FIXED_CLASSES)
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.empty(self.num_classes, self.num_classes))
        self._init_weights(rules)

    def _init_weights(self, rules: Optional[List[Tuple[int, int, float]]] = None):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        if rules:
            for t_from, t_to, score in rules:
                self.transitions.data[t_from, t_to] = score

    def forward(self, emissions, tags, mask=None):
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, num_classes) if batch_first is True,
                (seq_len, batch_size, num_classes) otherwise.
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            torch.Tensor: the (summed) log-likelihoods of each sequence in the batch.
                Shape of (1,)
        """

        # fix tensors order by setting batch as the first dimension
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)

    def decode(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, num_classes) if batch_first is True,
                (seq_len, batch_size, num_classes) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _compute_scores(self, emissions, tags, mask):
        """Compute the scores for a given batch of emissions with their tags.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, num_classes)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)

        # save first and last tags to be used later
        first_tags = tags[:, 1]
        last_valid_idx = mask.int().sum(1)
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add the transition from BOS to the first tags for each batch
        t_scores = self.transitions[self.BOS_TAG_IDX, first_tags]

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 1].gather(1, first_tags.unsqueeze(1)).squeeze()

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(2, seq_length):

            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster due to gpu,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add the transition from the end tag to the EOS tag for each batch
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]

        return scores

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, num_classes)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        _, seq_length, num_classes = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[self.BOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 1]

        for i in range(2, seq_length):
            alpha_t = []

            for tag in range(num_classes):

                # get the emission for the current tag
                e_scores = emissions[:, i, tag]

                # broadcast emission to all labels
                # since it will be the same for all previous tags
                # (bs, num_classes)
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag
                t_scores = self.transitions[:, tag]

                # broadcast the transition scores to all batches
                # (bs, num_classes)
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                # since alphas are in log space (see logsumexp below),
                # we add them instead of multiplying
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # create a torch matrix from alpha_t
            # (bs, num_classes)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return torch.logsumexp(end_scores, dim=1)

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, num_classes)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, num_classes = emissions.shape

        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.transitions[self.BOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 1]

        backpointers = []

        for i in range(2, seq_length):
            alpha_t = []
            backpointers_t = []

            for tag in range(num_classes):

                # get the emission for the current tag and broadcast to all labels
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag and broadcast to all batches
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # so far is exactly like the forward algorithm,
                # but now, instead of calculating the logsumexp,
                # we will find the highest score and the tag associated with it
                max_score, max_score_tag = torch.max(scores, dim=-1)

                # add the max score for the current tag
                alpha_t.append(max_score)

                # add the max_score_tag for our list of backpointers
                backpointers_t.append(max_score_tag)

            # create a torch matrix from alpha_t
            # (bs, num_classes)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            # append the new backpointers
            backpointers.append(backpointers_t)

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = torch.zeros(batch_size, seq_length, dtype=torch.long)
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].detach()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].detach()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)
            # add this path to the list of best sequences
            best_sequences[i][1 : sample_length + 1] = sample_path
        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.

            Args:
                sample_id (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, num_classes, batch_size) where seq_len_i
                represents the length of the ith sample in the batch

            Returns:
                list of ints: a list of tag indexes representing the bast path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].detach()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)
        return torch.tensor(best_path, dtype=torch.long)

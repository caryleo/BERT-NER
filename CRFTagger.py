from transformers.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from utils import argmax, log_sum_exp


class BertForCRFTagging(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForCRFTagging, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def load_option(self, opt):
        self.model = opt.model
        self.device = opt.device
        self.CRF=True
        if self.CRF:
            self.sigmoid = nn.Sigmoid()
            self.START_TAG = "<START>"
            self.STOP_TAG = "<STOP>"
            self.tag_to_ix = opt.tag2idx
            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            self.transitions = nn.Parameter(
                torch.randn(self.num_labels, self.num_labels))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -10000

    def _get_bert_features(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                           position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        # print("input_ids", input_ids.shape)
        # print("input_token_starts", input_token_starts.shape)
        # print("attention_mask", attention_mask.shape)
        # print("labels", labels.shape)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        # print("sequence_output", sequence_output.shape)

        #### 'X' label Issue Start ####
        # obtain original token representations from sub_words representations (by selecting the first sub_word)
        origin_sequence_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # print("padded_sequence_output", padded_sequence_output.shape)
        padded_sequence_output = self.dropout(padded_sequence_output)
        #### 'X' label Issue End ####

        logits = self.classifier(padded_sequence_output)
        if self.CRF:
            # logits = self.sigmoid(logits)  # 输出是概率
            logits = logits.squeeze(0)
        return logits

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.num_labels), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.num_labels):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.num_labels)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                           position_ids=None, inputs_embeds=None, head_mask=None):
        feats = self._get_bert_features(input_data, token_type_ids, attention_mask, labels,
                                        position_ids, inputs_embeds, head_mask)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, labels)
        return forward_score - gold_score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.num_labels), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.num_labels):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        logits = self._get_bert_features(input_data, token_type_ids, attention_mask, labels,
                                         position_ids, inputs_embeds, head_mask)
        if self.CRF:
            # logits = self.sigmoid(logits)  # 输出是概率
            score, label_seq_ids = self._viterbi_decode(logits)
            return score, label_seq_ids
        else:
            outputs = (logits,)
            if labels is not None:
                loss_mask = labels.gt(-1)
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if loss_mask is not None:
                    active_loss = loss_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), scores

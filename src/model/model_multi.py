# model += Parsing Infor Collecting Layer (PIC)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import BertModel, RobertaModel

import transformers
if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_bert import BertPreTrainedModel
    from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    from transformers.models.bert.modeling_bert import BertPreTrainedModel
    from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

from src.functions.biattention import BiAttention, BiLinear


class KorSciBERTForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, max_sentence_length, path):
        super(KorSciBERTForSequenceClassification, self).__init__(config, max_sentence_length, path)
        self.num_labels = config.num_labels
        self.num_coarse_labels = 3
        self.config = config

        self.bert = transformers.BertModel.from_pretrained(path, config=self.config)

        # special token <WORD>추가
        self.config.vocab_size = 15330 + 1
        self.bert.resize_token_embeddings(self.config.vocab_size)

        # 입력 토큰에서 token1, token2가 있을 때 (index of token1, index of token2)를 하나의 span으로 보고 이에 대한 정보를 학습
        self.span_info_collect = SICModel1(config)
        #self.span_info_collect = SICModel2(config)

        # biaffine을 통해 premise와 hypothesis span에 대한 정보를 결합후 정규화
        self.parsing_info_collect = PICModel(config, max_sentence_length)

        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else 0.1
        )

        # 대분류
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_coarse_labels)

        # 세부분류
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier2 = nn.Linear(config.hidden_size+self.num_coarse_labels, self.num_labels)

        self.reset_parameters #self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        coarse_labels=None,
        span=None,
        word_idxs=None,
    ):
        batch_size = input_ids.shape[0]
        discriminator_hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(sequence_output, word_idxs)

        # parser info collecting layer(PIC)
        hidden_states = self.parsing_info_collect(h_ij,
                                           batch_size= batch_size,
                                            span=span,)

        # 대분류
        hidden_states1 = self.dropout1(hidden_states)
        logits1 = self.classifier1(hidden_states1)

        # concat
        concat_hidden_states = torch.cat((logits1, hidden_states), dim=1)

        # 세부 분류
        hidden_states2 = self.dropout2(concat_hidden_states)
        logits2 = self.classifier2(hidden_states2)

        #logits = logits1
        logits = [logits1, logits2]
        outputs = (logits, ) + discriminator_hidden_states[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss1 = loss_fct(logits1.view(-1), coarse_labels.view(-1))
                loss2 = loss_fct(logits2.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            
            loss = loss1 + loss2
            # print("loss: "+str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def reset_parameters(self):
        self.dropout1.reset_parameters()
        self.classifier1.reset_parameters()
        self.dropout2.reset_parameters()
        self.classifier2.reset_parameters()

class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, max_sentence_length):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_coarse_labels = 3
        self.config = config
        self.bert = BertModel(config)

        # 입력 토큰에서 token1, token2가 있을 때 (index of token1, index of token2)를 하나의 span으로 보고 이에 대한 정보를 학습
        self.span_info_collect = SICModel1(config)
        #self.span_info_collect = SICModel2(config)

        # biaffine을 통해 premise와 hypothesis span에 대한 정보를 결합후 정규화
        self.parsing_info_collect = PICModel(config, max_sentence_length) # 구묶음 + tag 정보 + bert-biaffine attention + bilistm + bert-bilinear classification

        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else 0.1
        )
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_coarse_labels)
        self.classifier2 = nn.Linear(config.hidden_size+self.num_coarse_labels, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        coarse_labels=None,
        span=None,
        word_idxs=None,
    ):
        batch_size = input_ids.shape[0]
        discriminator_hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(sequence_output, word_idxs)

        # parser info collecting layer(PIC)
        hidden_states = self.parsing_info_collect(h_ij,
                                           batch_size= batch_size,
                                            span=span,)

        # 대분류
        hidden_states1 = self.dropout1(hidden_states)
        logits1 = self.classifier1(hidden_states1)

        # concat
        concat_hidden_states = torch.cat((logits1, hidden_states), dim=1)

        # 세부 분류
        hidden_states2 = self.dropout2(concat_hidden_states)
        logits2 = self.classifier2(hidden_states2)

        logits = [logits1, logits2]
        outputs = (logits, ) + discriminator_hidden_states[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss1 = loss_fct(logits1.view(-1), coarse_labels.view(-1))
                loss2 = loss_fct(logits2.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            loss = loss1+loss2
            #print("loss: "+str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, max_sentence_length):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_coarse_labels = 3
        self.config = config
        self.roberta = RobertaModel(config)

        # 입력 토큰에서 token1, token2가 있을 때 (index of token1, index of token2)를 하나의 span으로 보고 이에 대한 정보를 학습
        self.span_info_collect = SICModel1(config)
        #self.span_info_collect = SICModel2(config)

        # biaffine을 통해 premise와 hypothesis span에 대한 정보를 결합후 정규화
        self.parsing_info_collect = PICModel(config, max_sentence_length) # 구묶음 + tag 정보 + bert-biaffine attention + bilistm + bert-bilinear classification

        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else 0.1
        )
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_coarse_labels)
        self.classifier2 = nn.Linear(config.hidden_size+self.num_coarse_labels, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        coarse_labels=None,
        span=None,
        word_idxs=None,
    ):
        batch_size = input_ids.shape[0]
        discriminator_hidden_states = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(sequence_output, word_idxs)

        # parser info collecting layer(PIC)
        hidden_states = self.parsing_info_collect(h_ij,
                                           batch_size= batch_size,
                                            span=span,)

        # 대분류
        hidden_states1 = self.dropout1(hidden_states)
        logits1 = self.classifier1(hidden_states1)

        # concat
        concat_hidden_states = torch.cat((logits1, hidden_states), dim=1)

        # 세부 분류
        hidden_states2 = self.dropout2(concat_hidden_states)
        logits2 = self.classifier2(hidden_states2)

        logits = [logits1, logits2]
        outputs = (logits, ) + discriminator_hidden_states[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss1 = loss_fct(logits1.view(-1), coarse_labels.view(-1))
                loss2 = loss_fct(logits2.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            loss = loss1+loss2
            #print("loss: "+str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class SICModel1(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, word_idxs):
        # (batch, max_pre_sen, seq_len) @ (batch, seq_len, hidden) = (batch, max_pre_sen, hidden)
        word_idxs = word_idxs.squeeze(1)

        sen = torch.matmul(word_idxs, hidden_states)

        return sen

class SICModel2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_4 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, word_idxs):
        word_idxs = word_idxs.squeeze(1).type(torch.LongTensor).to("cuda")

        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb=torch.tensor([], dtype=torch.long).to("cuda")
        W2_hi_emb=torch.tensor([], dtype=torch.long).to("cuda")
        W3_hi_start_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W3_hi_end_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W4_hi_start_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W4_hi_end_emb = torch.tensor([], dtype=torch.long).to("cuda")
        for i in range(0, hidden_states.shape[0]):
            sub_W1_hi_emb = torch.index_select(W1_h[i], 0, word_idxs[i][0])  # (max_seq_length, hidden_size)
            sub_W2_hi_emb = torch.index_select(W2_h[i], 0, word_idxs[i][1])
            sub_W3_hi_start_emb = torch.index_select(W3_h[i], 0, word_idxs[i][0])
            sub_W3_hi_end_emb = torch.index_select(W3_h[i], 0, word_idxs[i][1])
            sub_W4_hi_start_emb = torch.index_select(W4_h[i], 0, word_idxs[i][0])
            sub_W4_hi_end_emb = torch.index_select(W4_h[i], 0, word_idxs[i][1])

            W1_hi_emb = torch.cat((W1_hi_emb, sub_W1_hi_emb.unsqueeze(0)))
            W2_hi_emb = torch.cat((W2_hi_emb, sub_W2_hi_emb.unsqueeze(0)))
            W3_hi_start_emb = torch.cat((W3_hi_start_emb, sub_W3_hi_start_emb.unsqueeze(0)))
            W3_hi_end_emb = torch.cat((W3_hi_end_emb, sub_W3_hi_end_emb.unsqueeze(0)))
            W4_hi_start_emb = torch.cat((W4_hi_start_emb, sub_W4_hi_start_emb.unsqueeze(0)))
            W4_hi_end_emb = torch.cat((W4_hi_end_emb, sub_W4_hi_end_emb.unsqueeze(0)))

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        span = W1_hi_emb + W2_hi_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hi_start_emb, W4_hi_end_emb) # (batch_size, max_seq_length, hidden_size)
        h_ij = torch.tanh(span)

        return h_ij


class PICModel(nn.Module):
    def __init__(self, config, max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_sentence_length = max_sentence_length

        # 구문구조 종류
        depend2idx = {"None": 0};
        idx2depend = {0: "None"};
        for depend1 in ['IP', 'AP', 'DP', 'VP', 'VNP', 'S', 'R', 'NP', 'L', 'X']:
            for depend2 in ['CMP', 'MOD', 'SBJ', 'AJT', 'CNJ', 'OBJ', "UNDEF"]:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2
        self.depend2idx = depend2idx
        self.idx2depend = idx2depend
        self.depend_embedding = nn.Embedding(len(idx2depend), self.hidden_size, padding_idx=0).to("cuda")

        self.reduction1 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction2 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))

        self.biaffine = BiAttention(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)

        self.bi_lism = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

    def forward(self, hidden_states, batch_size, span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        # span: (batch, max_prem_len, 3) -> (batch, max_prem_len, 3*hidden_size)
        new_span = torch.tensor([], dtype=torch.long).to("cuda")

        for i, span in enumerate(span.tolist()):
            span_head = torch.tensor([span[0] for span in span]).to("cuda") #(max_prem_len)
            span_tail = torch.tensor([span[1] for span in span]).to("cuda")
            span_dep = torch.tensor([span[2] for span in span]).to("cuda")

            span_head = torch.index_select(hidden_states[i], 0, span_head) #(max_prem_len, hidden_size)
            span_tail = torch.index_select(hidden_states[i], 0, span_tail)
            span_dep = self.depend_embedding(span_dep)

            n_span = span_head + span_tail + span_dep
            new_span = torch.cat((new_span, n_span.unsqueeze(0)))

        span = new_span
        del new_span

        # biaffine attention
        # hidden_states: (batch_size, max_prem_len, hidden_size)
        # span: (batch, max_prem_len, hidden_size)
        # -> biaffine_outputs: [batch_size, 100, max_prem_len,  max_prem_len]
        span = self.reduction1(span)
        hidden_states = self.reduction2(hidden_states)

        biaffine_outputs= self.biaffine(hidden_states, span)

        # bilstm
        # biaffine_outputs: [batch_size, 100, max_prem_len,  max_prem_len] -> [batch_size, 100, max_prem_len] -> [max_prem_len, batch_size, 100]
        # -> hidden_states: [batch_size, max_sentence_length]
        biaffine_outputs = biaffine_outputs.mean(-1)

        biaffine_outputs = biaffine_outputs.transpose(1,2).transpose(0,1)
        states = None

        bilstm_outputs, states = self.bi_lism(biaffine_outputs)

        hidden_states = states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        return hidden_states

    def reset_parameters(self):
        self.W_1_bilinear.reset_parameters()
        self.W_1_linear.reset_parameters()
        self.W_2_bilinear.reset_parameters()
        self.W_2_linear.reset_parameters()

        self.biaffine_W_bilinear.reset_parameters()
        self.biaffine_W_linear.reset_parameters()



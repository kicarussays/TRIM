from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import (BertForMaskedLM, BertForPreTraining,
                          BertForSequenceClassification)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM, MaskedLMOutput, CrossEntropyLoss, BertModel,
    BaseModelOutputWithPoolingAndCrossAttentions, MSELoss,
    MaskedLMOutput,
    BertEncoder, BertPooler, SequenceClassifierOutput, 
    SequenceClassifierOutput,
    MSELoss, BCEWithLogitsLoss, BertForPreTrainingOutput, BertLayer,
    BertAttention, BertIntermediate, BertOutput,
    )
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.embeddings import EhrEmbeddings
from src.adapter import AdapterEmbeddedBertEncoder, BottleneckAdapter


class BertModelLabelAdded(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
 
        self.embeddings = EhrEmbeddings(config)
        if config.problem_type == 'single_label_classification' and config.use_adapter:
            self.encoder = AdapterEmbeddedBertEncoder(config)
        else:
            self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if segment_ids is None:
            segment_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if record_rank_ids is None:
            record_rank_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if domain_ids is None:
            domain_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BehrtForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelLabelAdded(config, add_pooling_layer=False)
        self.representation = self.bert.embeddings.concept_embeddings
        
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        representations = self.representation(input_ids)

        masked_lm_loss = None
        if target is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss_mse = MSELoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target.view(-1))
            # smoothing_loss = loss_mse(sequence_output, representations)
            # if self.config.smooth:
            #     masked_lm_loss += 0.1 * smoothing_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ResNet(nn.Module):
    def __init__(self, conv_dim=8, fc_dim=768, lead=12, last_node=768, length=10):
        super().__init__()
        self.reduce1 = nn.Sequential(
            nn.Conv1d(lead, lead*conv_dim, kernel_size=11, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.reduce2 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=7, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.reduce3 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=5, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock1 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim)
        )
        
        self.activation = nn.ReLU()

        self.reduce4 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim*2, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(lead*conv_dim*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock2 = nn.Sequential(
            nn.Conv1d(lead*conv_dim*2, lead*conv_dim*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*2),
            nn.ReLU(),
            nn.Conv1d(lead*conv_dim*2, lead*conv_dim*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*2)
        )
        
        self.reduce5 = nn.Sequential(
            nn.Conv1d(lead*conv_dim*2, lead*conv_dim*4, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(lead*conv_dim*4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock3 = nn.Sequential(
            nn.Conv1d(lead*conv_dim*4, lead*conv_dim*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*4),
            nn.ReLU(),
            nn.Conv1d(lead*conv_dim*4, lead*conv_dim*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*4)
        )

        self.demo_layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.shared1 = nn.Linear(int(64*length*lead) + 64, fc_dim)
        self.shared1_1 = nn.Linear(int(64*length*lead), fc_dim)
        self.shared2 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc3 = nn.Linear(fc_dim, last_node)
        self.bothlayer = nn.Sequential(
            self.shared1, self.shared2, nn.ReLU(),
            self.fc2, self.bn2, nn.ReLU(),
            self.fc3)
        self.ecgonlylayer = nn.Sequential(
            self.shared1_1, self.shared2, nn.ReLU(),
            self.fc2, self.bn2, nn.ReLU(),
            self.fc3)
        
        self.lastdemo = nn.Linear(64, last_node)
        
        
    def forward(self, ecg, demo=None, which='ecg'):
        if which == 'demo':
            demo = self.demo_layer(demo)
            return self.lastdemo(demo)

        else:
            out = self.reduce1(ecg)
            out = self.reduce2(out)
            out = self.reduce3(out)
            
            out = self.activation(self.resblock1(out) + out)
            out = self.activation(self.resblock1(out) + out)
            
            out = self.reduce4(out)
            
            out = self.activation(self.resblock2(out) + out)
            out = self.activation(self.resblock2(out) + out)
            
            out = self.reduce5(out)
            
            out = self.activation(self.resblock3(out) + out)
            out = self.activation(self.resblock3(out) + out)
            
            out = out.view(out.size(0), -1)

            if which == 'ecg': 
                return self.ecgonlylayer(out)
            else:
                demo = self.demo_layer(demo)
                out = torch.cat((out, demo), dim=1)
                return self.bothlayer(out)



class EHRECGForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelLabelAdded(config, add_pooling_layer=True)
        self.resnet = ResNet()
        
        self.clf_ehr = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.clf_signal = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.clf_comb = nn.Linear(config.hidden_size*2, int(config.hidden_size / 2))
        self.clf_final = nn.Linear(config.hidden_size, config.num_labels)
        self.clf_single = nn.Linear(int(config.hidden_size / 4), config.num_labels)
        self.relu = nn.ReLU()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        signal: Optional[torch.Tensor] = None,
        return_embeddings: Optional[bool] = False,  # 추가
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        _pooled_output = None
        if self.config.ehr:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                age_ids=age_ids,
                segment_ids=segment_ids,
                record_rank_ids=record_rank_ids,
                domain_ids=domain_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            _pooled_output = outputs[1]
            pooled_output = self.dropout(_pooled_output)
            emb_ehr = self.relu(self.clf_ehr(pooled_output))

        if self.config.signal:
            outputs_ecg = self.resnet(signal.type(torch.float32))
            outputs_ecg = self.dropout(outputs_ecg)
            emb_ecg = self.relu(self.clf_signal(outputs_ecg))
        
        if self.config.ehr and self.config.signal:
            emb_comb = self.relu(self.clf_comb(torch.cat([pooled_output, outputs_ecg], dim=1)))
            logits = self.clf_final(torch.cat([emb_ehr, emb_comb, emb_ecg], dim=1))
        elif self.config.ehr:
            logits = self.clf_single(emb_ehr)
        else:
            logits = self.clf_single(emb_ecg)
            
        if return_embeddings:
            return torch.cat([pooled_output, outputs_ecg], dim=1)

        loss = None
        if target is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (target.dtype == torch.long or target.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), target.squeeze())
                else:
                    loss = loss_fct(logits, target)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), target.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, target.float())
        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=_pooled_output, # Pooled output for feature
            attentions=outputs.attentions,
        )

class ForSHAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clf_ehr = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.clf_signal = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.clf_comb = nn.Linear(config.hidden_size*2, int(config.hidden_size / 2))
        self.clf_final = nn.Linear(config.hidden_size, config.num_labels)
        self.clf_single = nn.Linear(int(config.hidden_size / 4), config.num_labels)
        self.relu = nn.ReLU()


    def forward(
        self,
        input
    ):
        pooled_output, outputs_ecg = input[:, :self.config.hidden_size], input[:, self.config.hidden_size:]
        emb_ehr = self.relu(self.clf_ehr(pooled_output))
        emb_ecg = self.relu(self.clf_signal(outputs_ecg))
        emb_comb = self.relu(self.clf_comb(torch.cat([pooled_output, outputs_ecg], dim=1)))
        logits = self.clf_final(torch.cat([emb_ehr, emb_comb, emb_ecg], dim=1))
        return logits

from transformers import AutoConfig
from packaging import version
import torch.nn as nn
from typing import Optional, Tuple, Union
import torch
import numpy as np

from transformers.modeling_outputs import (
    TokenClassifierOutput,
    SequenceClassifierOutputWithPast, 
    CausalLMOutputWithCrossAttentions)
# if version.parse(transformers.__version__) == version.parse('3.0.2'):
#     from transformers.modeling_gpt2 import GPT2ForSequenceClassification
# else: # transformers: version 4.0
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2ForSequenceClassification, GPT2LMHeadModel, 
    GPT2Model, GPT2PreTrainedModel, GPT2Config,
    GPT2ForTokenClassification)
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerModel
from transformers import AutoformerConfig, AutoformerForPrediction
import copy
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math

import warnings
warnings.filterwarnings('once') 


class PlasmaTransformerSeqtoLab(GPT2ForSequenceClassification):
    """GPT2ForSequenceClassification with a few modifications."""
    def __init__(
            self,
            n_head,
            n_layer,
            n_inner,
            activation_function,
            attn_pdrop,
            resid_pdrop,
            embd_pdrop,
            layer_norm_epsilon,
            pretrained_model,
            n_embd,
            max_length,
            attention_head_at_end,
            *args,
            **kwargs):
        # self.config = get_config(kwargs=kwargs)
        transformer_config = AutoConfig.from_pretrained('gpt2')
        transformer_config.n_head = n_head # self.config.num_sent_attn_heads
        transformer_config.n_layer = n_layer # self.config.num_contextual_layers
        transformer_config.n_inner = n_inner
        transformer_config.activation_function = activation_function
        transformer_config.attn_pdrop = attn_pdrop
        transformer_config.n_embd = n_embd # self.config.hidden_dim
        transformer_config.resid_pdrop = resid_pdrop
        transformer_config.n_positions = max_length # self.config.max_num_sentences + 20 # timestep window to classify over # may need to assign anything above 100 to the 100
        transformer_config.embd_pdrop = embd_pdrop
        transformer_config.n_ctx = transformer_config.n_positions
        transformer_config.pad_token_id = -100
        transformer_config.layer_norm_epsilon = layer_norm_epsilon
        
        super().__init__(transformer_config)
        self.pretrained_model = pretrained_model
        self.attention_head_at_end = attention_head_at_end
        self.self_att_end = AttentionCompression(n_embd, attn_pdrop)
        self.score_head = nn.Linear(n_embd, self.num_labels, bias=False)       

    def from_pretrained_lm(self, pretrained_model):
        # Copy weights from pretrained language model
        self.transformer = copy.deepcopy(pretrained_model.transformer)
        # Initialize a new sequence classification head
        self.score = nn.Linear(self.config.n_embd, self.config.num_labels)

    def process_one_shot(self, inputs_embeds, attention_mask, labels):
        # Process one-shot learning example

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states.squeeze()
        else: 
            # hidden states is 1 x 1 x hidden_dim and we only want the first dim gone
            hidden_states = hidden_states[0, :, :]
        last_hidden_states = self.self_att_end(hidden_states)
        logits = self.score_head(last_hidden_states)[0]
        loss = None
        if labels is not None: 
            loss = CrossEntropyLoss()(logits, labels)
        return loss, logits

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if not self.attention_head_at_end:
            return super().forward(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)

        losses = []
        pooled_logits = []

        if len(inputs_embeds.shape) == 3: 
            b = inputs_embeds.shape[0]
        else:
            b = 1
            inputs_embeds = inputs_embeds.unsqueeze(0)
        
        for i in range(b):
            if attention_mask is not None:
                am = attention_mask[i, :].unsqueeze(0)
                input_embeds = inputs_embeds[i, :, :].unsqueeze(0)
            else:
                input_embeds = inputs_embeds[i, :, :]
                mask = (input_embeds != -100.).any(dim=1).squeeze()
                input_embeds = input_embeds[mask, :]
                if input_embeds.shape[0] > 1:
                    input_embeds = input_embeds.unsqueeze(0)
                am = None
            if labels is not None:
                label = labels[i, :]
            else:
                label = None
            loss, logits = self.process_one_shot(input_embeds, am, label)
            losses.append(loss)
            pooled_logits.append(logits)

        #  out = torch.cumsum(out, dim=-1) # over the time. cumulative sum of all the model outputs so far. 
        #  out = out / torch.arange(1, out.shape[-1] + 1, device=out.device) # computing a running average. length independent. 
        #  out = self.out_layer(out) # dense 
        
        loss = None if losses[0] == None else sum(losses)
        pooled_logits = torch.vstack(pooled_logits)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class PlasmaTransformerSeqtoSeq(GPT2ForTokenClassification):
    """GPT2ForSequenceClassification with a few modifications."""
    def __init__(
            self,
            n_head,
            n_layer,
            n_inner,
            activation_function,
            attn_pdrop,
            resid_pdrop,
            embd_pdrop,
            layer_norm_epsilon,
            n_embd,
            max_length,
            attention_head_at_end,
            *args,
            **kwargs):
        # self.config = get_config(kwargs=kwargs)
        transformer_config = AutoConfig.from_pretrained('gpt2')
        transformer_config.n_layer = n_layer # self.config.num_contextual_layers
        transformer_config.num_attention_heads = n_head
        transformer_config.n_inner = n_inner
        transformer_config.activation_function = activation_function
        transformer_config.attn_pdrop = attn_pdrop
        transformer_config.n_embd = n_embd # self.config.hidden_dim
        transformer_config.resid_pdrop = resid_pdrop
        transformer_config.n_positions = max_length
        transformer_config.embd_pdrop = embd_pdrop
        transformer_config.n_ctx = transformer_config.n_positions
        transformer_config.pad_token_id = -100
        transformer_config.layer_norm_epsilon = layer_norm_epsilon
        transformer_config.num_labels = 2  # For binary classification
        
        super().__init__(transformer_config)    

        self.attention_head_at_end = False # attention_head_at_end
        self.self_att_end = AttentionCompression(n_embd, attn_pdrop)

    def from_pretrained_lm(self, pretrained_model):
        # Copy weights from pretrained language model
        self.transformer = copy.deepcopy(pretrained_model.transformer)

        # Initialize a new sequence classification head
        self.classifier = nn.Linear(self.config.n_embd, 2) 

        return

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)

        if self.attention_head_at_end:
            hidden_states = self.self_att_end(hidden_states)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class StatePredictionPlasmaTransformer(GPT2LMHeadModel):
    """GPT2ForSequenceClassification with a few modifications."""
    def __init__(
            self,
            n_head,
            n_layer,
            n_inner,
            activation_function,
            attn_pdrop,
            resid_pdrop,
            embd_pdrop,
            layer_norm_epsilon,
            loss,
            n_embd,
            max_length,
            attention_head_at_end,
            *args,
            **kwargs):
        # self.config = get_config(kwargs=kwargs)
        transformer_config = AutoConfig.from_pretrained('gpt2')
        transformer_config.n_layer = n_layer # self.config.num_contextual_layers
        transformer_config.num_attention_heads = n_head
        transformer_config.n_inner = n_inner
        transformer_config.activation_function = activation_function
        transformer_config.attn_pdrop = attn_pdrop
        transformer_config.n_embd = n_embd # self.config.hidden_dim
        transformer_config.resid_pdrop = resid_pdrop
        transformer_config.n_positions = max_length # self.config.max_num_sentences + 20 # timestep window to classify over # may need to assign anything above 100 to the 100
        transformer_config.embd_pdrop = embd_pdrop
        transformer_config.n_ctx = transformer_config.n_positions
        transformer_config.pad_token_id = -100
        transformer_config.layer_norm_epsilon = layer_norm_epsilon
        # transformer_config.num_labels = 2  # For binary classification
        transformer_config.vocab_size = n_embd # number of output features
        # transformer_config.output_hidden_states = True # not necessary 

        super().__init__(transformer_config)

        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "log_cosh":
            self.loss = LogCoshLoss()
        else:
            raise ValueError("select either mse or log_cosh for loss.")

        self.attention_head_at_end = attention_head_at_end
        self.self_att_end = AttentionCompression(n_embd, attn_pdrop)

        # replace 50,257 (vocab size) with 2! (binary classification)
        # self.lm_head = nn.Linear(transformer_config.n_embd, transformer_config.n_embd)  

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        mask = labels != -100

        loss = None

        if labels is not None:
            # Remove the last token from the logits
            shift_logits = lm_logits[..., :-1, :].contiguous().float()
            shift_labels = labels.contiguous().float()

            shift_logits_masked = shift_logits[mask]
            shift_labels_masked = shift_labels[mask]

            # Flatten the tokens
            loss_fct = self.loss
            loss = loss_fct(shift_logits_masked, shift_labels_masked)
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1, shift_labels.size(-1)))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
    

class AdditiveSelfAttention(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, input_dim)
        self.ws2 = nn.Linear(input_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ws2.state_dict()['weight'])

    def forward(self, hidden_embeds, context_mask=None):
        self_attention = torch.tanh(self.ws1(self.drop(hidden_embeds)))
        # self attention : if one doc: (num sentences in curr batch x max_len x hidden_dim
                                                                              #   if >1 doc: if many docs: (num docs x num sents x max word len x hidden_dim)
        self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)      # self_attention : (num_sentences in curr batch x max_len)
        if context_mask is not None:
            context_mask = -10000 * (context_mask == 0).float()
            self_attention = self_attention + context_mask                    # self_attention : (num_sentences in curr batch x max_len)
        if len(self_attention.shape) == 1:
            self_attention = self_attention.unsqueeze(0)  # todo: does this cause problems?
        self_attention = self.softmax(self_attention).unsqueeze(1)            # self_attention : (num_sentences in curr batch x 1 x max_len)
        return self_attention


class AttentionCompression(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=hidden_size, dropout=dropout)

    def forward(self, hidden_embs, attention_mask=None):
        ## `'hidden_emds'`: shape = N x hidden_dim
        self_attention = self.self_attention(hidden_embs, attention_mask)  # self_attention = N x 1 x N
        ## batched matrix x batched matrix:
        output_encoding = torch.matmul(self_attention, hidden_embs).squeeze(1)
        return output_encoding


class PlasmaAutoformer(AutoformerForPrediction):
    def __init__(
        self,
        prediction_length,
        context_length,
        num_time_features,
        num_static_categorical_features,
        num_static_real_features,
        num_real_dynamic_features,
        prediction_input_feature_size,
        num_parallel_samples,
        lags_sequence,
        cardinality,
        static_cat_embedding_dim,
    ):
        
        autoformer_config = AutoformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            num_time_features=num_time_features,
            num_static_categorical_features=num_static_categorical_features,
            num_static_real_features=num_static_real_features,
            input_size=prediction_input_feature_size,
            num_real_dynamic_features=num_real_dynamic_features,
            num_parallel_samples=num_parallel_samples,
            lags_sequence=lags_sequence,
            cardinality=cardinality, 
            embedding_dimension=static_cat_embedding_dim,
        )

        super().__init__(autoformer_config) 
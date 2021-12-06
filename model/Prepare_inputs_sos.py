from transformers import BertLMHeadModel


class SOSSwinBert(BertLMHeadModel):
    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):

        input_shape=input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        if past is None:
             output_dict ={
                      "inputs_embeds" :model_kwargs['inputs_embeds'],
                      "attention_mask": attention_mask,
                      "past_key_values": past,
                      "encoder_hidden_states": model_kwargs['encoder_hidden_states']}
        else:
            output_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past,
                "encoder_hidden_states": model_kwargs['encoder_hidden_states']}

        return output_dict




import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from transformers import Trainer
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer import get_tpu_sampler


try:
    from .utils import label_smoothed_nll_loss
except ImportError:
    from utils import label_smoothed_nll_loss


logger = logging.getLogger(__name__)


class Seq2SeqTrainer(Trainer):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [
            '<|HOME|>', '<|AWAY|>',
            '<|PLAYER-START_POSITION|>', '<|PLAYER-MIN|>', '<|PLAYER-PTS|>', '<|PLAYER-FGM|>', '<|PLAYER-FGA|>', '<|PLAYER-FG_PCT|>', '<|PLAYER-FG3M|>', '<|PLAYER-FG3A|>', '<|PLAYER-FG3_PCT|>', '<|PLAYER-FTM|>', '<|PLAYER-FTA|>', '<|PLAYER-FT_PCT|>', '<|PLAYER-OREB|>', '<|PLAYER-DREB|>', '<|PLAYER-REB|>', '<|PLAYER-AST|>', '<|PLAYER-TO|>', '<|PLAYER-STL|>', '<|PLAYER-BLK|>', '<|PLAYER-PF|>', 
            '<|TEAM-PTS_QTR1|>', '<|TEAM-PTS_QTR2|>', '<|TEAM-PTS_QTR3|>', '<|TEAM-PTS_QTR4|>', '<|TEAM-PTS|>', '<|TEAM-FG_PCT|>', '<|TEAM-FG3_PCT|>', '<|TEAM-FT_PCT|>', '<|TEAM-REB|>', '<|TEAM-AST|>', '<|TEAM-TOV|>', '<|TEAM-WINS|>', '<|TEAM-LOSSES|>', '<|TEAM-CITY|>', '<|TEAM-NAME|>', 
        ]})
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.get_freq_sequences(data_dir)
        self.seq_loss_weight = 2
    
    def get_freq_sequences(self, data_dir):
        big_map = defaultdict(int)
        with open(os.path.join(data_dir, "train.target"), 'r', encoding='utf-8') as f:
            for paragraph in f.readlines():
                words = paragraph.split(' ')
                for i in range(0, len(words)-2):
                    li = words[i:i+3]
                    has_num = False
                    for tok in li:
                        num = ''
                        try:
                            num = int(tok)
                        except:
                            try:
                                num = text2num(tok)
                            except:
                                pass
                        if isinstance(num, int):
                            has_num = True
                    if not has_num:
                        current_seq = ' '.join(li)
                        big_map[current_seq] += 1
        tokens = self.tokenizer.batch_encode_plus(
            [k for k, v in sorted(big_map.items(), key=lambda item: item[1], reverse=True)][:75], 
            return_tensors='pt',
            pad_to_max_length=True
        )
        self.freq_seq = {tuple(x[1:3].tolist()): x[4] for x in tokens['input_ids']}
        print(self.freq_seq)
    
    def _trigram_penalty(self, output):
        with torch.no_grad():
            penalty_factor = torch.ones(len(output))
            # get output of current step
            for batch in range(len(output)):
                penalty_factor[batch] += 1
                probs = torch.argmax(output[batch], dim=-1)#.view(-1, self.config.vocab_size), dim=-1)
                for i in range(0, len(probs)-2):
                    li = tuple(probs[i:i+2].tolist())
                    if li in self.freq_seq:
                        if self.freq_seq[li] != probs[i+2]:
                            penalty_factor[batch] += 1

            penalty = torch.mean(penalty_factor)
        return torch.log(penalty) if penalty > 1 else torch.log(penalty)*self.seq_loss_weight
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size, distributed=self.args.n_gpu > 1
                )

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs[0]
        return self._compute_loss(logits, labels, ignore_index=model.config.pad_token_id)

    def _compute_loss(self, logits, labels, ignore_index):
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            assert logits.shape[-1] == self.model.config.vocab_size
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=ignore_index
            )
        loss += self._trigram_penalty(logits)
        return loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        max_length = (
            model.config.max_generate_length
            if hasattr(model.config, "max_generate_length")
            else model.config.max_position_embeddings
        )

        with torch.no_grad():
            if self.args.predict_with_generate and not self.args.prediction_loss_only:
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    num_beams=model.config.num_beams,
                    max_length=max_length,
                )
                # in case the batch is shorter than max length, the output should be padded
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, max_length, model.config.pad_token_id
                )

            labels_out = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs[1]
            loss = self._compute_loss(logits, labels_out, model.config.pad_token_id)
            loss = loss.mean().item()
            if self.args.prediction_loss_only:
                logits = None
            else:
                logits = generated_tokens if self.args.predict_with_generate else logits

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels_out = labels_out.detach()
        labels = self._pad_tensors_to_max_len(labels_out, max_length, model.config.pad_token_id)
        return (loss, logits.detach(), labels)

    def _pad_tensors_to_max_len(self, tensor, max_length, pad_token_id):
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

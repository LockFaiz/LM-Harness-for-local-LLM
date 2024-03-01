import os
import copy
from tqdm import tqdm
# from pathlib import Path
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import utils
from typing import List, Literal, Optional, Tuple, Union
from lm_eval.api.instance import Instance
from lm_eval.utils import Collator, stop_sequences_criteria
import torch


eval_logger = utils.eval_logger

def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)

@register_model("mistral-7b")
class Mistral(LM):
    # AUTO_MODEL_CLASS = None
    REQ_CHUNK_SIZE = 8
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self, 
        model, 
        tokenizer, 
        *, 
        max_tokens: int =35, 
        temperature: float = 0.7,
        top_p: float = 0.8, 
        max_length: Optional[int] = None,
        Buffer = None,
        token_chunk: int = None,
        accel: bool = False
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.n_words
        self.end_of_text_token_id = self.tokenizer.eos_id
        self._max_gen_toks = max_tokens
        self._max_length = max_length
        self._batch_size = model.args.max_batch_size # number of requests to process at once
        self.Buffer = Buffer
        self.token_chunk = token_chunk
        self._device = self.model.device
        self.temperature = temperature
        self.top_p = top_p

        # accelerator: unfinished
        if accel:
            try:
                from accelerate import Accelerator
            except ModuleNotFoundError:
                raise Exception("accelerate module not found")
            gpus = torch.cuda.device_count()
            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                self.accelerator = accelerator
            if not (parallelize or accelerator.num_processes > 1):
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        # if str(batch_size).startswith("auto"):
        #     batch_size = batch_size.split(":")
        #     self.batch_size_per_gpu = batch_size[0]
        #     self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        # else:
        #     self.batch_size_per_gpu = int(batch_size)
        
        # multi gpu or single gpu
        self._rank = 0
        self._world_size = 1
        


    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        else:
            return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        if self._batch_size:
            return self._batch_size
        else:
            return self.REQ_CHUNK_SIZE
    @property
    def device(self):
        return self._device

    # @property
    # def batch_size(self):
    #     return self.batch_size_per_gpu

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    def tok_encode(self, string: str, bos=False) -> List[int]:
        return self.tokenizer.encode(string, bos=bos)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # whole_enc = self.tokenizer.encode(context + continuation, bos=False)
        # context_enc = self.tokenizer.encode(context, bos=False)
        # print(f'context:{context}\ncontinuation:{continuation}')
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc


    def cache(self, seqlens):
        cache_window = max(seqlens) + self.max_gen_toks #17+35=52
        if self.model.args.sliding_window is not None and cache_window > self.model.args.sliding_window: # 4096
            cache_window = model.args.sliding_window
        cache = self.Buffer(
            self.model.n_local_layers,
            self.model.args.max_batch_size,
            cache_window,
            self.model.args.n_kv_heads,
            self.model.args.head_dim,
        )
        cache.to(device=self.model.device, dtype=self.model.dtype)
        cache.reset()
        return cache

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            # if len(new_reqs) == 3:
            #     break
            # print('context:', context)
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.end_of_text_token_id],
                    self.tok_encode(continuation),
                )
                # assert context != "" 'context is empty! '
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                # context_enc = [self.tokenizer.encode(prompt, bos=True) for prompt in [context]]
                # continuation_enc = [self.tokenizer.encode(prompt, bos=True) for prompt in [continuation]]
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        print(f'Number of Requests(loglikelihood): {len(new_reqs)}')
        return self._loglikelihood_tokens(new_reqs)


    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        loglikelihoods = []
        adaptive_batch_size = None

        for (string,) in tqdm([req.args for req in requests], disable=(self.rank != 0)):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            # pad_amnt = 0
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )
            string_nll = [x[0] for x in string_nll]            
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        print(f'loglikelihood_rolling: {len(loglikelihoods)}')
        return loglikelihoods

    @torch.inference_mode()
    def generate(
        self, 
        prompts_tokens: List[List[int]], 
        ipnlens: List[int],
        *, 
        generate_token: bool = False,
        max_generate_token: int = 35,
        until = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        ):
        self.model = self.model.eval()
        B, V = len(prompts_tokens), self.model.args.vocab_size
        cache = self.cache(ipnlens)

        logprobs = [[] for _ in range(B)] # [[], [], []]
        last_token_prelogits = None
        max_prompt_len = max(ipnlens)
        if self.token_chunk is None:
            self.token_chunk = max_prompt_len
        
        for s in range(0, max_prompt_len, self.token_chunk):
            prompt_chunks = [p[s:s+self.token_chunk] for p in prompts_tokens]
            # assert all(len(p) > 0 for p in prompt_chunks), "mistral_eval.generate: prompt_chunks is empty!"
            # assert all(len(p) > 0 for p in prompt_chunks) 
            prelogits = self.model.forward(
                torch.tensor(sum(prompt_chunks, []), device=self.model.device, dtype=torch.long),
                seqlens=[len(p) for p in prompt_chunks],
                cache=cache
            )
            logits = torch.log_softmax(prelogits, dim=-1)
            if last_token_prelogits is not None:
                last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
                for i_seq in range(B):
                    # logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())
                    logprobs[i_seq].append(last_token_logits[i_seq].tolist())

            offset= 0
            for i_seq, sequence in enumerate(prompt_chunks):
                # logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
                logprobs[i_seq].extend([logits[offset + i].tolist() for i in range(len(sequence) - 1)])
                offset += len(sequence)
            
            last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)
            assert last_token_prelogits.shape == (B, V)
        if not generate_token:
            return logprobs
        generated_tokens = [[] for _ in range(B)]
        assert last_token_prelogits is not None
        assert isinstance(until, list), 'You should provide proper stopping string'
        for last_idx in range(len(last_token_prelogits)):
            for _ in range(max_generate_token):
                next_token = sample(last_token_prelogits[last_idx].unsqueeze(0), temperature=temperature, top_p=top_p)
                assert isinstance(next_token, torch.Tensor) and len(next_token) == 1, 'Next_token is not a tensor or its length is not 1'
                last_token_logits = torch.log_softmax(last_token_prelogits[last_idx].unsqueeze(0), dim=-1)
                # for i in range(B):
                #     logprobs[i].append(last_token_logits[i, next_token[i]].item())
                assert last_token_logits.shape == (1, V), 'last_token_logits is not right shape'
                # logprobs[last_idx].append(last_token_logits[0, next_token[0]].item())
                logprobs[last_idx].append(last_token_logits[0])
                loop = True
                generated_tokens[last_idx].extend(next_token)
                for stop in until:
                    assert isinstance(stop, str), 'Stopping criteria should be a string'
                    generate_string = self.tok_decode(generated_tokens[last_idx].tolist())
                    assert isinstance(generate_string, str), 'generate_string is not a string'
                    if generate_string.endswith(stop):
                        loop = False
                    else:
                        last_token_prelogits[last_idx] = self.model.forward(next_token, seqlens=[1], cache=cache)
                        assert last_token_prelogits.shape == (B, V), 'last_token_prelogits is not right shape'
                if not loop:
                    continue
        generated_words = []
        assert generated_tokens, 'generated_tokens is empty'
        assert generated_tokens
        for i, x in enumerate(prompts_tokens):
            generated_words.append(self.tok_decode(generated_tokens[i].tolist()))
        return generated_words, logprobs


            
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # re_ord = Collator(requests, sort_fn=_collate)
        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            inps = []
            cont_toks_list = []
            inplens = []
            for _, context_enc, continuation_enc in chunk:
                # inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                inp = context_enc + continuation_enc
                inps.append(inp)  # [[ctx1+ctn1],...,[ctxn+ctnn],] length=self.chunk_size
                inplens.append(len(inp))
                cont_toks_list.append(continuation_enc)
            chunk_logprobs = self.generate(inps, inplens)
            
            # label_token_list = []
            # for label_token in cont_toks_list:
            #     if label_token not in label_token_list:
            #         label_token_list.append(label_token)
            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, chunk_logprobs, inplens, cont_toks_list
            ):
                # print(f'all logits length:{len(logits)}; vocub_size:{len(logits[0])}')
                logits = torch.tensor(logits, device=self.model.device)
                logits = torch.log_softmax(logits, dim=-1)
                contlen = len(cont_toks)
                # print(f'all logits shape:{logits.shape}')
                # print(f'continuation length: {contlen}')
                cont_logits = logits[-contlen:].unsqueeze(0) # [1, seq, vocab]
                # print('+' *10)
                # print(f'cont_logits shape:{cont_logits.shape}')
                # logits torch.Size([3398]) cont_logits shape:torch.Size([1, 1]); cont_toks shape:1
                # print('+' *10)
                # print(f'cont_logits after softmax:{cont_logits}')
                # print(f'cont_logits after softmax shape:{cont_logits.shape}')
                greedy_tokens = cont_logits.argmax(dim=-1) # [1, seq]
                cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.model.device).unsqueeze(0) # [1, seq]
                # print(f'greedy_tokens:{greedy_tokens}, cont_toks:{cont_toks}')
                max_equal = (greedy_tokens == cont_toks).all()
                # print(f'cont_logits:{cont_logits.shape}, cont_toks:{cont_toks.shape}')
                cont_logits = torch.gather(cont_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [1, seq]

                answer = (float(cont_logits.sum()), bool(max_equal))
                res.append(answer)
                self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        print(f'result of _loglikelihood_tokens: {res}')
        return re_ord.get_original(res)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        # re_ord = utils.Reorderer(requests, _collate)
        re_ords = Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(m=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                if 'until' in kwargs.keys():
                    until = kwargs.pop('until')
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}")
            else:
                raise ValueError(f"Expected `kwargs` to be of type `dict` but got {kwargs}")
            if not until:
                until = [self.tok_decode(self.eot_token_id)]   
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks  

            max_ctx_len = self.max_length - max_gen_toks

            # max_ctx_len = self.max_length - max_gen_toks 
            context_enc = torch.tensor([self.tok_encode(p, bos=True) for p in contexts], dtype=torch.long, device=self.model.device)
            context_enc = context_enc[:, -max_ctx_len:]
            ctx_token_string = [self.tok_decode(p) for p in context_enc]
            ctx_token_string_len = [len(p) for p in ctx_token_string]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks
            cont_str, logprob = self.generate(
                ctx_token_string, 
                ctx_token_string_len, 
                generate_token=True, 
                max_generate_token=max_gen_toks, 
                until=until,
                temperature=self.temperature,
                top_p = self.top_p
            )
            for s, context in zip(cont_str, contexts):
                res.append(cont_str)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res





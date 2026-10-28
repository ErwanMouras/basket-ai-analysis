"""Local PARSeq scorebug OCR: GPU first, using the jersey reader's CPU fallback."""

import math

from training.jersey.reader import CHARSET, ParseqReader


class ScorebugReader(ParseqReader):
    def __init__(self, config, *, manage_threads=True):
        super().__init__(config, manage_threads=manage_threads)
        self.provenance.update(model="PARSeq scorebug", alphabet=CHARSET,
                               max_decode_length=7,
                               decoding="full-vocabulary autoregressive; EOS and confidence required")

    def _read_numbers(self, batch, torch):
        model = self.model
        steps = 8
        memory = model.encode(batch)
        queries = model.pos_queries[:, :steps].expand(batch.shape[0], -1, -1)
        mask = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=batch.device), 1)
        context = torch.full((batch.shape[0], steps), 96, dtype=torch.long, device=batch.device)
        context[:, 0] = 95
        tokens, probabilities = [], []
        for index in range(steps):
            output = model.decode(context[:, :index + 1], memory, mask[:index + 1, :index + 1],
                                  tgt_query=queries[:, index:index + 1],
                                  tgt_query_mask=mask[index:index + 1, :index + 1])
            scores = model.head(output[:, 0]).softmax(-1)
            probability, token = scores.max(-1)
            tokens.append(token)
            probabilities.append(probability)
            if index + 1 < steps:
                context[:, index + 1] = token
        ids = torch.stack(tokens, 1).cpu().tolist()
        confidence = torch.stack(probabilities, 1).cpu().tolist()
        results = []
        for row, scores in zip(ids, confidence):
            end = row.index(0) if 0 in row else len(row)
            results.append({"text": "".join(CHARSET[t - 1] for t in row[:end]),
                            "confidence": math.prod(scores[:end + 1]), "eos": end < len(row)})
        return results

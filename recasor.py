from recasepunc import init, Config, Model, case, punctuation, WordpieceTokenizer, mapped_punctuation
from recasepunc import recase, punctuation_syms
import torch

REV_CASE = {b: a for a, b in case.items()}
REV_PUNC = {b: a for a, b in punctuation.items()}

class Recasor(object):
    def __init__(self):
        config = Config(device="cpu")
        checkpoint_path = "checkpoint/es.24000"
        loaded = torch.load(checkpoint_path, map_location=config.device if torch.cuda.is_available() else 'cpu')
        keys_not_needed = ["iteration", "optimizer_state_dict", "train_loss", "valid_loss",
        "valid_accuracy_case", "valid_accuracy_punc", "valid_fscore"]
        for key in keys_not_needed:
            if key in loaded:
                del loaded[key]
        if 'config' in loaded:
            self.config = Config(**loaded['config'])
            init(self.config)
        new_state_dict = {}
        original_state_dict = loaded['model_state_dict']
        for key, value in original_state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        self.model = Model(self.config.flavor, self.config.device)
        self.model.load_state_dict(new_state_dict)
        del loaded['model_state_dict']
        del new_state_dict

    def predict(self, text):
        prediction = ""
        for line in text.split("\n"):
            # also drop punctuation that we may generate
            line = ''.join([c for c in line if c not in mapped_punctuation])
            if self.config.debug:
                print(line)
            tokens = [self.config.cls_token] + self.config.tokenizer.tokenize(line) + [self.config.sep_token]
            if self.config.debug:
                print(tokens)
            previous_label = punctuation['PERIOD']
            first_time = True
            was_word = False
            for start in range(0, len(tokens), self.config.max_length):
                instance = tokens[start: start + self.config.max_length]
                ids = self.config.tokenizer.convert_tokens_to_ids(instance)
                # print(len(ids), file=sys.stderr)
                if len(ids) < self.config.max_length:
                    ids += [self.config.pad_token_id] * (self.config.max_length - len(ids))
                x = torch.tensor([ids]).long().to(self.config.device)
                y_scores1, y_scores2 = self.model(x)
                y_pred1 = torch.max(y_scores1, 2)[1]
                y_pred2 = torch.max(y_scores2, 2)[1]
                for id, token, punc_label, case_label in zip(ids, instance, y_pred1[0].tolist()[:len(instance)],
                                                            y_pred2[0].tolist()[:len(instance)]):
                    if self.config.debug:
                        print(id, token, punc_label, case_label, file=sys.stderr)
                    if id in (self.config.cls_token_id, self.config.sep_token_id):
                        continue
                    if previous_label is not None and previous_label > 1:
                        if case_label in [case['LOWER'], case['OTHER']]:
                            case_label = case['CAPITALIZE']
                    previous_label = punc_label
                    # different strategy due to sub-lexical token encoding in Flaubert
                    if self.config.lang == 'fr':
                        if token.endswith('</w>'):
                            cased_token = recase(token[:-4], case_label)
                            if was_word:
                                print(' ', end='')
                            print(cased_token + punctuation_syms[punc_label], end='')
                            was_word = True
                        else:
                            cased_token = recase(token, case_label)
                            if was_word:
                                print(' ', end='')
                            print(cased_token, end='')
                            was_word = False
                    else:
                        if token.startswith('##'):
                            cased_token = recase(token[2:], case_label)
                            # print(cased_token, end='')
                            prediction += cased_token
                        else:
                            cased_token = recase(token, case_label)
                            if not first_time:
                                prediction += ' '
                                # print(' ', end='')
                            first_time = False
                            # print(cased_token + punctuation_syms[punc_label], end='')
                            prediction += cased_token + punctuation_syms[punc_label]
            if previous_label == 0:
                print('.', end='')
                prediction += '.'
            print()
        return prediction
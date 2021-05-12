# -*- coding: UTF-8 -*-​ 
import json
import collections
import unicodedata
import torch

torch.manual_seed(512)

# 参考google团队
# http://www.apache.org/licenses/LICENSE-2.0

class InputExample:
    def __init__(self, guid, context, candidate, label):
        self.guid = guid
        self.context = context
        self.candidate = candidate
        self.label = label

class DataProcessor:
    def __init__(self, data_path):
        self.datasets = dict()
        for data_type in data_path.keys():
            with open(data_path[data_type], "r", encoding="utf-8") as f:
                data = json.load(f)
                self.datasets[data_type] = data

    def _convert_to_unicode(self, text):
        """将输入转为unicode编码"""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def _is_whitespace(self, char):
        """判断字符是否为空格"""
        # 将'\t'、'\n'、'\r'当作空格，用于将文本切分为token
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs": # Separator, Space
            return True
        return False

    def _is_control(self, char):
        """判断字符是否为控制字符"""
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"): # Control
            return True
        return False

    def _is_punctuation(self, char):
        """判断字符是否为标点符号"""
        cp = ord(char)
        # 不在unicode中的字符，如： "^", "$"和"`" 用SSCII判断
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"): # Punctuation
            return True
        return False
    
    def _clean_text(self, text):
        """删除无效字符, 将\t\r\n等字符用空格替代"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _whitespace_tokenize(self, text):
        """用空格将文本进行切分"""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens
    
    def _strip_accents(self, token):
        """去除token中的重音"""
        token = unicodedata.normalize("NFD", token)
        output = []
        for char in token:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _split_on_punc(self, token):
        """对token根据标点符号进行再切分，并将标点符号作为一个单独的token"""
        chars = list(token)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _create_examples(self, datas, data_type):
        """将输入的json封装成input_example数据结构"""
        examples = []
        for idx, data in enumerate(datas):
            # if idx > 10:
            #     break
            contexts = []
            candidates = []
            for text in data["messages-so-far"]:
                contexts.append(text["utterance"])
            gt = data["options-for-correct-answers"][0]["candidate-id"]
            # print(len(data["options-for-next"]))
            for candidate_idx, candidate in enumerate(data["options-for-next"]):
                candidates.append(candidate["utterance"])
                if candidate["candidate-id"] == gt:
                    label = candidate_idx
            guid = "%s-%s" % (data_type, data["example-id"])
            examples.append(
                InputExample(guid=guid, context=contexts, candidate=candidates, label=label)
            )
        return examples

    def get_train_examples(self):
        return self._create_examples(self.datasets["train"], "train")
    
    def get_dev_examples(self):
        return self._create_examples(self.datasets["dev"], "dev")
    
    def get_dataset_tokens(self, examples):
        """将文本切分成token列表"""
        datasets = []
        for example in examples:
            guid = example.guid
            contexts = example.context
            candidates = example.candidate
            label = example.label

            contexts_tokens = []
            for context in contexts:
                context = self._convert_to_unicode(context)
                context = self._clean_text(context)
                tokens = self._whitespace_tokenize(context)
                post_tokens = []
                for token in tokens:
                    token = token.lower()
                    token = self._strip_accents(token)
                    post_tokens.extend(self._split_on_punc(token))
                contexts_tokens.append(post_tokens)
                
            candidates_tokens = []
            for candidate in candidates:
                candidate = self._convert_to_unicode(candidate)
                candidate = self._clean_text(candidate)
                tokens = self._whitespace_tokenize(candidate)
                post_tokens = []
                for token in tokens:
                    token = token.lower()
                    token = self._strip_accents(token)
                    post_tokens.extend(self._split_on_punc(token))
                candidates_tokens.append(post_tokens)

            datasets.append(
                InputExample(guid=guid, context=contexts_tokens, candidate=candidates_tokens, label=label)
            )
        return datasets

    def create_vocab(self, datasets, vocab_path):
        """用训练集的token创建词表"""
        count_dict = dict()
        for dataset in datasets:
            contexts = dataset.context
            candidates = dataset.candidate

            for context in contexts:
                for token in context:
                    if token not in count_dict:
                        count_dict[token] = 0
                    count_dict[token] += 1

            for candidate in candidates:
                for token in candidate:
                    if token not in count_dict:
                        count_dict[token] = 0
                    count_dict[token] = 1

        token_count_sorted = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
        
        with open(vocab_path, "w") as f:
            f.write("<pad>\n") # 句子填充
            f.write("<unk>\n") # 代表词表中未出现的词
            for item in token_count_sorted:
                f.write(item[0] + "\n")
    
    def _load_vocab(self, vocab_path, vocab_size):
        token2index = dict()
        with open(vocab_path, "r", encoding="utf-8") as f:
            idx = 0
            for token in f.readlines():
                token = self._convert_to_unicode(token)
                token = token.strip()
                token2index[token] = idx
                idx += 1
                if idx > vocab_size:
                    break
        return token2index, min(idx, vocab_size)
    
    def get_dataset_indices(self, datasets, vocab_path, vocab_size):
        """用字典将token转为index"""
        vocab, vocab_size = self._load_vocab(vocab_path, vocab_size)
        dataset_indices = []
        for dataset in datasets:
            guid = dataset.guid
            contexts = dataset.context
            candidates = dataset.candidate
            label = dataset.label   

            context_indices = []
            for context in contexts:
                indices = []
                for token in context:
                    if token in vocab:
                        indices.append(vocab[token])
                    else:
                        indices.append(vocab["<unk>"])
                context_indices.append(indices)
            
            candidate_indices = []
            for candidate in candidates:
                indices = []
                for token in context:
                    if token in vocab:
                        indices.append(vocab[token])
                    else:
                        indices.append(vocab["<unk>"])
                candidate_indices.append(indices)
            dataset_indices.append(
                InputExample(guid=guid, context=context_indices, candidate=candidate_indices, label=label)
            )
        return dataset_indices, vocab_size
        
    def create_tensor_dataset(self, datas, max_turn_num, max_seq_len):
        """创建数据集"""
        all_contexts = []
        all_candidates = []
        all_labels = []
        for data in datas:
            contexts = data.context
            candidates = data.candidate
            label = data.label

            new_contexts = []
            if len(contexts) > max_turn_num:
                contexts = contexts[-max_turn_num:]
            # 对话轮数不足的用全0<pad>补
            contexts += [[0] * max_seq_len] * (max_turn_num - len(contexts))
            for context in contexts:
                if len(context) > max_seq_len:
                    context = context[-max_seq_len:]
                context += [0] * (max_seq_len - len(context)) #短文本末尾0<pad>填充
                new_contexts.append(context)
            
            new_candidates = []
            for candidate in candidates:
                if len(candidate) > max_seq_len:
                    candidate = candidate[-max_seq_len:]
                candidate += [0] * (max_seq_len - len(candidate))
                new_candidates.append(candidate)
            
            all_contexts.append(new_contexts)
            all_candidates.append(new_candidates)
            all_labels.append(label)

        all_contexts = torch.LongTensor(all_contexts)
        all_candidates = torch.LongTensor(all_candidates)
        all_labels = torch.LongTensor(all_labels)

        tensor_dataset = torch.utils.data.TensorDataset(all_contexts, all_candidates, all_labels)
        return tensor_dataset
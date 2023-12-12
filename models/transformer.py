import math
from math import sqrt

import numpy as np
from torch import nn
import torch

class SDPA(nn.Module):
    def __init__(self, cfg):
        super(SDPA, self).__init__()
        self.cfg = cfg
        self.dk = cfg.dmodel // cfg.h

        # TODO: инициализация Pytorch softmax
        self.softmax = nn.Softmax(dim=-1)
        # TODO: инициализация dropout
        self.dropout = nn.Dropout()


    def forward(self, Q, K, V, mask_self_attention=None, mask_padding=None):
        """
            Вычисляет SDPA.
            Формула: SDPA(Q, K, V) = softmax((QK^T) / sqrt(dk))V
            QK^T - матричное умножение query и key, K^T - транспонирование key.
            Масштабирующий множитель sqrt(dk) нормализует скалярные произведения.

        Args:
            Q (torch.Tensor): Тензор queries. Размерность  [batch_size, l, dk],
                              где seq_len - длина последовательности queries, dk - размерность векторов запросов.
            K (torch.Tensor): Тензор keys. Размерность  [batch_size, n, dk].
            V (torch.Tensor): Тензор values. Размерность  [batch_size, n, dk],
                              где dk - размерность векторов values.
            mask_self_attention (torch.Tensor): Тензор bools, маска для self attention декодера.
                                                Размерность  [batch_size, n, l].
            mask_padding (torch.Tensor): Тензор bools, маска для padding.
                                                Размерность  [batch_size, n, l].

        Returns:
            torch.Tensor: Тензор, представляющий взвешенное суммирование values, взвешенное
                          коэффициентами внимания, полученными после применения механизма SDPA к Q, K и V.
                          Размерность выходного тензора обычно [batch_size, l, dv].

        """

        # 1. Расчет скалярных произведений query (q) и key (k),
        #    деление каждого на sqrt(dk) для масштабирования.
        #    dk - размерность векторов key и query.
        #    Получаем необработанные оценки внимания.
        # TODO: написать код для получения необработанных оценок внимания
        sdpa = torch.mm(Q, torch.t(K)) / (self.dk ** 0.5)

        # 1.1 Если mask_self_attention и/или mask_padding не None, заполнить необработанные оценки внимания
        # значениями -inf в тех местах, где mask_self_attention и/или mask_padding True
        # TODO: написать код для обработки масок
        if mask_self_attention is not None:
            sdpa = torch.masked_fill(sdpa, mask_self_attention, float('-inf'))

        if mask_padding is not None:
            sdpa = torch.masked_fill(sdpa, mask_padding, float('-inf'))


        # 2. Применение функции softmax к необработанным оценкам внимания для получения коэффициентов внимания.
        #    Шаг softmax гарантирует, что коэффициенты положительны и в сумме дают 1.
        # TODO: написать код с применением softmax к необработанным оценкам внимания
        sdpa = self.softmax(sdpa)

        # 3. Умножение коэффициентов внимания на матрицу values (V) и суммирование для получения итогового результата.
        #    Оператор @ здесь представляет собой пакетное матричное умножение коэффициентов внимания
        #    на тензор значений.
        #  TODO: написать код перемножения коэффициентов внимания на матрицу values
        sdpa = sdpa @ V

        # 3. Применение dropout
        #  TODO: написать код применения dropout
        sdpa = self.dropout(sdpa)
        return sdpa

class SHA(nn.Module):
    def __init__(self, cfg):
        super(SHA, self).__init__()
        self.cfg = cfg
        self.dk = cfg.dmodel // cfg.h

        # TODO: Инициализация линейных преобразований для Q, K, V
        self.weights_q = nn.Linear(cfg.dmodel, cfg.dmodel)
        self.weights_k = nn.Linear(cfg.dmodel, cfg.dmodel)
        self.weights_v = nn.Linear(cfg.dmodel, cfg.dmodel)

        # Инициализация механизма SDPA
        self.sdpa = SDPA(self.cfg)

    def forward(self, Q, K, V, mask_self_attention=None, mask_padding=None):
        """
            Вычисляет SHA.
            Формула: SHA(Q, K, V) = SDPA(Q', K', V')
            Q', K', V' - линейно преобразованные тензоры Q, K, V.

        Args:
            Q (torch.Tensor): Тензор queries.
            K (torch.Tensor): Тензор keys.
            V (torch.Tensor): Тензор values.
            mask_self_attention (torch.Tensor): Тензор bools, маска для self attention декодера.
            mask_padding (torch.Tensor): Тензор bools, маска для encoder-decoder attention декодера.

        Returns:
            torch.Tensor: Взвешенное суммирование values, взвешенное коэффициентами внимания.

        """

        # TODO: Линейные преобразования Q, K, V

        Q_transformed = self.weights_q(Q)
        K_transformed = self.weights_k(K)
        V_transformed = self.weights_v(V)

        # TODO: Вызов SDPA с преобразованными Q, K, V

        output = self.sdpa(Q_transformed, K_transformed, V_transformed, mask_self_attention, mask_padding)

        return output

class MHA(nn.Module):
    def __init__(self, cfg):
        super(MHA, self).__init__()
        self.cfg = cfg

        # Инициализация списка SHA модулей
        self.sha_list = nn.ModuleList([SHA(cfg) for _ in range(cfg.h)])

        # TODO: Инициализация линейного преобразования для объединения выходов из всех головок внимания
        self.weights_o = nn.Linear(cfg.dmodel * cfg.h, cfg.dmodel)

    def forward(self, Q, K, V, mask_self_attention=None, mask_padding=None):
        """
            Вычисляет MHA.
            Формула: MHA(q, k, v) = Concat(SHA1, SHA2, ..., SHAh)W^O
            где SHAi - выход i-го Single Head Attention, W^O - линейное преобразование.

        Args:
            Q (torch.Tensor): Тензор queries.
            K (torch.Tensor): Тензор keys.
            V (torch.Tensor): Тензор values.
            mask_self_attention (torch.Tensor): Тензор bools, маска для self attention декодера.
            mask_padding (torch.Tensor): Тензор bools, маска для encoder-decoder attention декодера.

        Returns:
            torch.Tensor: Результат Multi-Head Attention.

        """

        # TODO: Вычисление выходов для каждого SHA

        sha_outputs = [sha(Q, K, V, mask_self_attention, mask_padding) for sha in self.sha_list]

        # TODO: Конкатенация выходов и применение линейного преобразования

        concatenated_output = torch.cat(sha_outputs, dim=-1)

        output = self.weights_o(concatenated_output)

        return output

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.cfg = cfg

        # Первый линейный слой увеличивает размерность данных с dmodel до 4*dmodel.
        self.w1 = nn.Linear(cfg.dmodel, 4 * cfg.dmodel)
        # Второй линейный слой уменьшает размерность обратно с 4*dmodel до dmodel.
        self.w2 = nn.Linear(4 * cfg.dmodel, cfg.dmodel)

        # Функция активации ReLU используется между двумя линейными слоями.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Формула: FF(x) = ReLU(xW1 + b1)W2 + b2
        где:
        - W1, b1 - веса и смещение первого линейного слоя,
        - W2, b2 - веса и смещение второго линейного слоя,

        Args:
            x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

        Returns:
            torch.Tensor: Выходной тензор с той же размерностью, что и входной.
        """
        x = self.w1(x)

        x = self.relu(x)

        x = self.w2(x)

        return x

class PositionEncoder(nn.Module):
    def __init__(self, cfg):
        super(PositionEncoder, self).__init__()
        self.cfg = cfg

        # Создание матрицы позиционного кодирования
        # Размер матрицы: [cfg.max_sentence_len, cfg.dmodel]
        self.pe_matrix = torch.empty((cfg.max_sentence_len, cfg.dmodel))

        # Формула для позиционного кодирования:
        # PE(pos, 2i) = sin(pos / (10000 ^ (2i / dmodel)))
        # PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / dmodel)))
        # где pos - позиция в предложении, i - индекс в векторе
        position = torch.arange(0, cfg.max_sentence_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, cfg.dmodel, 2).float() * -(math.log(10000.0) / cfg.dmodel))
        self.pe_matrix[:, 0::2] = torch.sin(position * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(position * div_term)
        # Полезно знать. Пусть a - numpy array. Тогда a[0::2] выдает элементы на четных позициях, а a[1::2] на нечетных.

        # Инициализация dropout
        self.dropout = nn.Dropout(cfg.dropout)


    def forward(self, x):
        """
       Прямой проход PositionEncoder. Добавляет positional encoding к входному тензору.

       Positional encoding вектор вычисляется как:
       PE(pos, 2i) = sin(pos / (10000 ^ (2i / dmodel)))
       PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / dmodel)))
       где pos - позиция в предложении, i - индекс в векторе.

       Args:
           x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

       Returns:
           torch.Tensor: Тензор с добавленным позиционным кодированием.
       """
        # Вычисление размера предложения из входного тензора
        seq_len = x.size(1)

        # Добавление позиционного кодирования к входному тензору
        x = x + self.pe_matrix[:seq_len, :].unsqueeze(0)

        # Использование dropout
        x = self.dropout(x)

        return x

class EncoderSingleLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderSingleLayer, self).__init__()
        # Инициализация Multi-Head Attention (MHA)
        self.mha = MHA(cfg)
        # Инициализация нормализации
        self.ln1 = nn.LayerNorm(cfg.dmodel)
        self.ln2 = nn.LayerNorm(cfg.dmodel)
        # Инициализация полносвязного Feed Forward слоя
        self.ff = FeedForward(cfg)

        # Инициализация 2-x dropout
        self.dropout_1 = nn.Dropout(cfg.dropout)
        self.dropout_2 = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """
        Прямой проход одного слоя энкодера.

        Этапы:
        1. Применение Multi-Head Attention.
        1.1 Применение dropout
        2. Добавление исходного входа к результату (Residual Connection).
        3. Применение Layer Normalization.
        4. Применение Feed Forward слоя.
        4.1 Применение dropout
        5. Добавление результата после MHA к результату FF (Residual Connection).
        6. Применение Layer Normalization.

        Args:
            x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

        Returns:
            torch.Tensor: Тензор после одного слоя энкодера.
        """
        # Применение MHA, добавление Residual Connection и Layer Normalization
        mha_output = self.mha(x, x, x)
        mha_output = self.dropout_1(mha_output)
        x_res1 = x + mha_output
        x_ln1 = self.ln1(x_res1)

        # Применение Feed Forward, добавление Residual Connection и Layer Normalization
        ff_output = self.ff(x_ln1)
        ff_output = self.dropout_2(ff_output)
        x_res2 = x_ln1 + ff_output
        x_ln2 = self.ln2(x_res2)

        return x_ln2

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        # Создание N слоев энкодера cfg.N
        self.seq = nn.ModuleList([EncoderSingleLayer(cfg) for _ in range(cfg.N)])
        self.cfg = cfg

    def forward(self, x):
        """
        Прямой проход через энкодер.

        Последовательно применяет N слоев энкодера к входным данным.

        Args:
            x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

        Returns:
            torch.Tensor: Тензор после прохождения через N слоев энкодера.
        """
        # Применение каждого слоя энкодера
        for layer in self.seq:
            x = layer(x)
        return x

class DecoderSingleLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderSingleLayer, self).__init__()

        # Инициализация Multi-Head Attention для self attention декодера
        self.self_attention = MHA(cfg)

        # Инициализация слоя нормализации и Dropout
        self.ln1 = nn.LayerNorm(cfg.dmodel)
        self.dropout_1 = nn.Dropout(cfg.dropout)

        # Инициализация Multi-Head Attention для внимания между энкодером и декодером
        self.encoder_decoder_attention = MHA(cfg)

        # Инициализация слоя нормализации и Dropout
        self.ln2 = nn.LayerNorm(cfg.dmodel)
        self.dropout_2 = nn.Dropout(cfg.dropout)

        # Инициализация полносвязного Feed Forward слоя
        self.ff = FeedForward(cfg)

        # Инициализация слоя нормализации и Dropout
        self.ln3 = nn.LayerNorm(cfg.dmodel)
        self.dropout_3 = nn.Dropout(cfg.dropout)

    def forward(self, x, enc_out, mask_for_pad_decoder, mask_for_pad_encoder_decoder, mask):
        """
        Forward pass одного слоя декодера.

        Этапы:
        1. Применение self-attention с маской предотвращения утечки будущей информации и маской для padding входа декодера.
        2. Добавление Dropout, residual connection и применение Layer Normalization.
        3. Применение attention между энкодером и декодером с маской для padding входа декодера и энкодера.
        4. Добавление Dropout, residual connection и применение Layer Normalization.
        5. Применение Feed Forward слоя.
        6. Добавление Dropout, residual connection и применение Layer Normalization.

        Args:
            x (torch.Tensor): Входной тензор декодера.
            enc_out (torch.Tensor): Выходной тензор энкодера.
            mask_for_pad_decoder (torch.Tensor): Маска padding входа энкодера.
            mask_for_pad_encoder_decoder (torch.Tensor): Маска padding входа декодера и энкодера.
            mask (torch.Tensor): Маска для предотвращения утечки будущей информации.

        Returns:
            torch.Tensor: Тензор после одного слоя декодера.
        """
        # Применение self-attention, residual connection, Layer Normalization и Dropout
        self_attention_output = self.self_attention(x, x, x, mask_for_pad_decoder, mask)
        self_attention_output = self.dropout_1(self_attention_output)
        x_res1 = x + self_attention_output
        x_ln1 = self.ln1(x_res1)

        # Применение attention между энкодером и декодером, residual connection, Layer Normalization и Dropout
        enc_dec_attention_output = self.encoder_decoder_attention(x_ln1, enc_out, enc_out, mask_for_pad_encoder_decoder, mask)
        enc_dec_attention_output = self.dropout_2(enc_dec_attention_output)
        x_res2 = x_ln1 + enc_dec_attention_output
        x_ln2 = self.ln2(x_res2)

        # Применение Feed Forward, residual connection, Layer Normalization и Dropout
        ff_output = self.ff(x_ln2)
        ff_output = self.dropout_3(ff_output)
        x_res3 = x_ln2 + ff_output
        x_ln3 = self.ln3(x_res3)

        return x_ln3

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        # Создание N слоев декодера
        self.seq = nn.ModuleList([DecoderSingleLayer(cfg) for _ in range(cfg.N)])
        self.cfg = cfg

    def forward(self, x, enc_out, mask_for_pad_decoder, mask_for_pad_encoder_decoder, mask):
        """
        Forward pass через декодер.

        Последовательно применяет N слоев декодера к входным данным.

        Args:
            x (torch.Tensor): Входной тензор декодера.
            enc_out (torch.Tensor): Выходной тензор энкодера.
            mask_for_pad_encoder (torch.Tensor): Маска padding входа декодера.
            mask_for_pad_encoder_decoder (torch.Tensor): Маска padding входа декодера и энкодера.
            mask (torch.Tensor): Маска для предотвращения утечки будущей информации.

        Returns:
            torch.Tensor: Тензор после прохождения через N слоев декодера.
        """
        # Применение каждого слоя декодера
        for layer in self.seq:
            x = layer(x, enc_out, mask_for_pad_decoder, mask_for_pad_encoder_decoder, mask)
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()

        # Инициализация слоя для получения эмбеддингов и позиционного кодирования
        # для задания с переводом вам нужно 2 embedding_layer
        self.embedding_layer = nn.Embedding(cfg.voc_size, cfg.dmodel)
        self.pe = PositionEncoder(cfg)


        # Инициализация энкодера
        self.encoder = Encoder(cfg)

        # Инициализация декодера
        self.decoder = Decoder(cfg)

        # Инициализация слоя линейного преобразования для выхода
        # Примечание: Веса выходного слоя могут быть связаны с весами для получения эмбеддингов
        # для задания с переводом берите веса с  embedding_layer для декодера
        self.output_projection_layer = nn.Linear(cfg.dmodel, cfg.voc_size)
        self.output_projection_layer.weight = self.embedding_layer.weight
        self.output_projection_layer.bias = None

    def forward(self, enc_inp, dec_inp, mask_for_pad_decoder, mask_for_pad_encoder_decoder, mask):
        """
        Forward pass через трансформер.

        Этапы:
        1. Применение слоя для получения эмбеддингов и позиционного кодирования к входным данным энкодера и декодера.
        2. Прохождение через энкодер.
        3. Прохождение через декодер.
        4. Применение выходного линейного преобразования.

        Args:
            enc_inp (torch.Tensor): Входные данные энкодера.
            dec_inp (torch.Tensor): Входные данные декодера.
            mask_for_pad_decoder (torch.Tensor): Маска паддинга для входных данных энкодера.
            mask_for_pad_encoder_decoder (torch.Tensor): Маска паддинга для входных данных декодера и энкодера.
            mask (torch.Tensor): Маска для предотвращения утечки будущей информации в декодере.

        Returns:
            torch.Tensor: Выходные данные трансформера.
        """
        # Применение слоя для получения эмбеддингов и позиционного кодирования
        enc_inp = self.embedding_layer(enc_inp)
        dec_inp = self.embedding_layer(dec_inp)
        enc_inp = self.pe(enc_inp)
        dec_inp = self.pe(dec_inp)

        # Прохождение через энкодер
        enc_out = self.encoder(enc_inp)

        # Прохождение через декодер
        dec_out = self.decoder(dec_inp, enc_out, mask_for_pad_decoder, mask_for_pad_encoder_decoder, mask)

        # Применение выходного линейного преобразования
        final_output = self.output_projection_layer(dec_out)

        return final_output


if __name__ == "__main__":
    from config.transformer_cfg import cfg
    q = torch.randn((1,5,cfg.dmodel // cfg.h))
    k = torch.randn((1,10,cfg.dmodel // cfg.h))
    v = torch.randn((1,10,20))

    sdpa = SDPA(cfg)
    output = sdpa(q,k,v)

    q = torch.randn((1, 5, cfg.dmodel))

    mha = MHA(cfg)
    output = mha(q, q, q)


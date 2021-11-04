import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
import torch.nn.functional as F
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}#adding cpi's code
class Net(nn.Module):
    def __init__(self, pretrained_dir, tag_num, lr = 5e-5, device = 'gpu', fineTune = False):#device = 'cpu'
        super().__init__()
        # Shared
        self.bert = BertModel.from_pretrained(pretrained_dir)        
        self.device = device
        self.fineTune = fineTune
        num_filters = 200
        kernel_sizes = [6, 8, 10]
        bert_hidden_size = self.bert.config.hidden_size#768
        self.dropout = nn.Dropout(0.5)
        # NER
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=bert_hidden_size, hidden_size=bert_hidden_size//2, batch_first=True)
        self.fc = nn.Linear(bert_hidden_size, tag_num)
        # RC
        self.convs_rc = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, bert_hidden_size)) for k in kernel_sizes])#add:org:bert_hidden_size#3* scheme3
        self.linear = nn.Linear(768*3, 2)##3*200,#len(kernel_sizes) * num_filters#768*3

        self.activation_final = nn.Tanh()  # 高斯
        self.activation = ACT2FN["gelu"]
        # Triage
        self.triageCls = nn.Linear(len(kernel_sizes) * num_filters, 2)
        # self.triageCls = nn.Linear(bert_hidden_size, 2)

        self.optimizer = optim.Adam([
            {'params': self.bert.parameters(), 'lr': 1e-5},
            {'params': self.lstm.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.linear.parameters()},
            {'params': self.triageCls.parameters()},
            {'params': self.convs_rc.parameters()},
            ], lr = lr
        )

        if device != 'cpu':
            self.cuda(device=self.device)

    def conv_and_pool(self, x, conv):
        x = conv(x.unsqueeze(1))#[4, 200, 507, 1]/[4, 200, 505, 1]/[4, 200, 503, 1]#concat:[4, 200, 507, 1537]
        # print("x 's shape is:", x.shape)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)#[4,200]
        # print("pool's shape is:", x.shape)
        return x

    def forwardNER(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        # with torch.no_grad():
        last_hidden_state, _ = self.bert(x)
        last_hidden_state, _ = self.lstm(last_hidden_state)
        logits = self.fc(last_hidden_state)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

    def forwardRC(self, x, y, P_gauss1_bs, P_gauss2_bs):
        """ Take a mini-batch of Examples, compute the probability of relation
        @param examples (List[InputExample]): list of InputExample

        @returns  loss
        """
        y = y.to(self.device)
        for k in x:
            x[k] = x[k].to(self.device)
        if self.fineTune:
            last_hidden_state, pooled_output = self.bert(**x)#last_hidden_state.shape[4, 512, 768]|pooled_output[4,768]
            # print("last_hidden_state's shape is:", last_hidden_state.shape)
            # print("pooled_output's shape is:", pooled_output.shape)

        else:
            with torch.no_grad():
                last_hidden_state, pooled_output = self.bert(**x)
        last_hidden_state = self.dropout(last_hidden_state)  #
        last_hidden_state = self.activation(last_hidden_state)
        P_gauss1_bs = P_gauss1_bs.to(self.device)
        P_gauss2_bs = P_gauss2_bs.to(self.device)#(4,512)
        # print("P_gauss1_bs.shape: ",P_gauss1_bs.shape)
        P_gauss1_bs = P_gauss1_bs.unsqueeze(dim=1)
        P_gauss2_bs = P_gauss2_bs.unsqueeze(dim=1)#(4,512)->#(4,1,512)
        gauss_entity1 = torch.matmul(P_gauss1_bs, last_hidden_state)
        gauss_entity2 = torch.matmul(P_gauss2_bs, last_hidden_state)#4,1,768

        # print("gauss_entity1 is:", gauss_entity1)
        gauss_entity1 = gauss_entity1.squeeze(dim=1)#4,768
        gauss_entity2 = gauss_entity2.squeeze(dim=1)
        # gauss_entity1 = gauss_entity1.repeat(1, 512, 1)#add
        # gauss_entity2 = gauss_entity2.repeat(1, 512, 1)#add
        gauss_entity1 = self.activation_final(gauss_entity1)
        gauss_entity2 = self.activation_final(gauss_entity2)
        gauss_entity1 = self.dropout(gauss_entity1)
        gauss_entity2 = self.dropout(gauss_entity2)
        #print("gauss_entity1's shape is:", gauss_entity1.shape)
        pooled_output = self.dropout(pooled_output)
        #print("self.convs_rc's shape is:", self.conv.shape)
        out = torch.cat((pooled_output, gauss_entity1, gauss_entity2), -1) #scheme 1 wrong  # 4,2304
        # out_cat = torch.cat((last_hidden_state, gauss_entity1, gauss_entity2), 1)#scheme 2 wrong #[4,512,768],[4,1,768],[4,1,768]

        # out_cat = torch.cat((last_hidden_state, gauss_entity1, gauss_entity2), -1)#scheme 3
        # out = torch.cat([self.conv_and_pool(out_cat, conv) for conv in self.convs_rc], 1)#[4,200*3]
        out = self.dropout(out)
        # print("out is:", out)

        logits = self.linear(out)
        y_hat = logits.argmax(-1)
        # print("logits is:", logits)
        # print("y_hat is:", y_hat)
        return logits, y, y_hat

    def forwardTriage(self, x, y):
        """ Take a mini-batch of Examples, compute the probability of relation
        @param examples (List[InputExample]): list of InputExample

        @returns  loss
        """
        y = y.to(self.device)
        for k in x:
            x[k] = x[k].to(self.device)
        if self.fineTune:
            last_hidden_state, pooled_output = self.bert(**x)
        else:
            with torch.no_grad():
                last_hidden_state, pooled_output = self.bert(**x)
        out = torch.cat([self.conv_and_pool(last_hidden_state, conv) for conv in self.convs_rc], 1)
        out = self.dropout(out)
        logits = self.triageCls(out)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

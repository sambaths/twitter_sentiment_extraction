import config
import transformers
import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Roberta(transformers.BertPreTrainedModel):
    def __init__(self,conf):
        super(Roberta, self).__init__(conf)
        self.config = conf
        self.model = transformers.RobertaModel.from_pretrained(config.MODEL_PATH, config = conf)
        self.bert_drop = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128,64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64,768, kernel_size=3, padding=1)
        self.l0 = nn.Linear(128,128)


    def forward(self, ids, mask, token_type_ids):
        if not self.config.output_hidden_states:
            o1, o2= self.model(
                ids, 
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
            print(o1.shape, o2.shape)
            out1 = self.bert_drop(o1.permute(0,2,1))
            print('After Dropout :',out1.shape)
            out1 = self.conv1(out1)
            print('After Conv1 :',out1.shape)
            out1 = nn.LeakyReLU()(out1)
            print('After Act1 :',out1.shape)
            out1 = self.conv2(out1)
            print('After Conv2 :',out1.shape)
            out1 = nn.LeakyReLU()(out1)
            out1 = self.conv3(out1)
            print('After Conv3 :',out1.shape)
            out1 = self.l0(out1)
            print('After Lin :',out1.shape)
            out1 = Flatten()(out1)
            print('After Flatten :',out1.shape)
            # out1 = torch.softmax(out1, dim=1)
            # print('After Act2 :',out1.shape)


            out2 = self.bert_drop(o1.permute(0,2,1))
            out2 = self.conv1(out2)
            out2 = nn.LeakyReLU()(out2)
            out2 = self.conv2(out2)
            out2 = nn.LeakyReLU()(out2)
            out2 = self.conv3(out2)
            out2 = self.l0(out2)
            out2 = Flatten()(out2)
            # out2 = torch.softmax(out2, dim=1)

        else:
            o1, o2, o3= self.model(
                ids, 
                attention_mask=mask,
                token_type_ids=token_type_ids
            )


            out1 = self.bert_drop(o1)
            out1 = self.conv1(out1)
            out1 = nn.LeakyReLU()(out1)
            out1 = self.conv2(out1)
            out1 = self.l0(out1)
            out1 = Flatten()(out1)
            out1 = nn.Softmax()(out1)


            out2 = self.bert_drop(o1)
            out2 = self.conv1(out2)
            out2 = nn.LeakyReLU()(out2)
            out2 = self.conv2(out2)
            out2 = self.l0(out2)
            out2 = Flatten()(out2)
            out2 = nn.Softmax()(out2)


        return out1, out2
        

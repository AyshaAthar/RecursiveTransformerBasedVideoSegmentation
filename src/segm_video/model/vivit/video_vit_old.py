import torch
import torch.nn as nn

import torch.nn.functional as F
import math


from einops import rearrange

from segm.model.vivit.attention import Block
from segm.model.vivit.vit import VisionTransformer
from segm.model.vivit.decoder import DecoderLinear



class VideoTransformer(nn.Module):
    def __init__(self,image_size,patch_size,num_images,n_layers,d_model,d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers =  n_layers
        self.num_images = num_images
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_cls = n_cls
        self.d_ff = d_ff
        super().__init__()
        
        self.vit_encoder = VisionTransformer(image_size, patch_size,n_layers,d_model,n_cls,n_heads,d_ff)

        self.temporal_token = nn.Parameter(torch.randn(1,4,1,d_model))
        
        dpr = [x.item() for x in torch.linspace(0, 0.0, 12)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, 0.1, dpr[i]) for i in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)
                
        self.decoder = DecoderLinear(n_cls,patch_size,d_model)
                
        
    def forward(self,images):
        
        segmented_output=[]
        
        b,t,c,h,w = images.shape
        num_seqs = images.shape[1]
        
        #images=images.reshape(4,1,3,768,768)
        
        images = rearrange(images, "b t c h w -> t b c h w")
        
        vit_encoderoutput=[]
        
        for i in range(num_seqs):
            print("------------------------------------------------------------------------")
            print("For image", i,"in sequence")
            x=self.vit_encoder(images[i])   #-> images[i]-> B*C*H*W
            
            if(i==0):
                decoder_input = x[:, 1:]
                decoder_output = self.decoder(decoder_input,(h,w))
                masks = F.interpolate(decoder_output, size=(768, 768), mode="bilinear")
                segmented_output.append(masks)
                vit_encoderoutput.append(x)
                vit_embds = torch.stack(vit_encoderoutput)
            else:
                #vit_embds = torch.stack(vit_encoderoutput)
                
                vit_embds = rearrange(vit_embds,"t b n d -> b t n d")
                #vit_embds = vit_embds.reshape(1,4,2305, 192)
                self.temporal_token = nn.Parameter(torch.randn(1,i,1,d_model))

                print("Temporal token",self.temporal_token.shape)
                #print(vit_embds.shape)

                vit_embds = self.temporal_token + vit_embds

                vit_embds = rearrange(vit_embds, "b t n d -> b (t n) d")
                #vit_embds = vit_embds.reshape(1,4*2305,192)
                print("Spatio-temporal Transformer input shape ->", vit_embds.shape)
                for blk in self.blocks:
                    vit_embds = blk(vit_embds)
                    
                #Saving the output from the spatio-temporal transformer to be used for the corrector transformer
                spatemp_embds = vit_embds
                
                print("Spatial-transformer output embeddinngs",spatemp_embds.shape)
                
                spatemp_embds = rearrange(spatemp_embds , "b (t n) d  -> b t n d",t=i)
                                
                #Discarding one embedding
                
                spatemp_embds = spatemp_embds[:,:1]
                
                #rearranging spatio-temporal embedding 
                
                spatemp_embds = rearrange(spatemp_embds, "b t n d -> b (t n) d")
                
                #Concatenating with vit_encoder output of current time step 
                
                corrector_input = torch.cat([spatemp_embds , x], dim=1)
                
                print("Corrector transformer Input -> ", corrector_input.shape )
                
                #Passing through corrector transformer
                for blk in self.blocks:
                    corrector_input = blk(corrector_input)
                    
                #After passing through transformer
                print("After passing through corrector transformer", corrector_input.shape)
                
                #After rearranging
                corrector_output = rearrange(corrector_input, " b (t n) d -> b t n d", t=2)
                
                
                #Discardng one output
                corrector_output = corrector_output[:,:1]
                
                print("After discarding pne embedding from corrector transformer", corrector_output.shape)
                
                #Combining batch & time
                corrector_output = rearrange(corrector_output,"b t n d -> (b t) n d")                
                
                corrector_output = self.norm(corrector_output)
                                
                #Decoder
                decoder_input = corrector_output
                decoder_input = decoder_input[:, 1:]
                
                decoder_output = self.decoder(decoder_input,(h,w))
                                
                masks = F.interpolate(decoder_output, size=(768, 768), mode="bilinear")
                segmented_output.append(masks)
                
                print("Final decoder output", masks.shape)
                
                #Appending results of the vit_encoder for the current time step, so it can be used in the later timsteps
                vit_encoderoutput.append(x)
                vit_embds = torch.stack(vit_encoderoutput)

        return segmented_output
        


if __name__ == "__main__":
    image_size = (768,768)
    patch_size = 16
    num_images = 4
    n_layers = 12
    d_model = 192
    d_ff = 4*192
    n_heads = 3
    n_cls = 19
    model = VideoTransformer(image_size, patch_size, num_images, n_layers, d_model, d_ff, n_heads, n_cls, d_ff)




    images = torch.rand(1,4,3,768,768)
    x = model(images)








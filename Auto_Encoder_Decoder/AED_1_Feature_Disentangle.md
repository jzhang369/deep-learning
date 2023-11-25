# Feature Disentangle

## Use voice conversion as an example

[Video](https://www.youtube.com/watch?v=Jj6blc8UijY)

![Voice Conversion Using Feature Disentangle](https://d3i71xaburhd42.cloudfront.net/4e1f36855442b761729dad4507513e23ca66206c/15-Figure10-1.png)

This system architecture shows how the content encoder, speaker encoder, and the decoder have been trained and ready to generate target speech. However, in the trainning phase, the source speaker speech, the target speaker speech, and the converted target speech should all be a speech. 



Designing Encoders Using Some Intuitive Approaches:
+ Speaker Encoder: You can use an existing speaker embedding (and perhaps fine-tune it) to encode the speaker's information. Examples include
  + i-vector
  + d-vector
  + x-vector
+ Content Encoder: 
  + You can directly use a speech recognition (e.g., speech -> text)
  + Train a content encoder so that the speaker classifer can be fooled. 


Designing Encoders By Devising a Network Architecture

+ instance normalization (remove speaker information)
+ 
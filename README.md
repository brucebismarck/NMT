# NMT
Neural Machine Translation with attention in both encoder and decoder


This is a project from Dr. Mohit NLP class's final project. The detail description was shown in the tex pdf.

Neural Network framework used in this NMT is Pytorch.

Feature:
1. Seq2Seq with 3layers biGRU  model as encoder and decoder
2. global attention in decoder
3. self attention in encoder
4. Alignment and visualization
5. Model Evaluation.


Files:
1. data_processing.py is used to read in data, select some of the features such as :
    select minlen of sentence
    select maxlen of sentence
    select if the sentence starts with some specific prefixes
    
2. embedding.py is used to do wordembedding. For small dataset such as the eng-fra.txt in this repository, embedding method does not provide significant improvement to the result.
     To use this embedding method, please download glove embedding and French embedding file which I downloaded from here
     https://github.com/Kyubyong/wordvectors
     Note: because I used glove embedding at the first time, and glove uses 300 dim embedding. Please also find 300 dim embedding 
     
     
3. model.py is used to define models. They are:
      basic Encoder
      Encoder with self attentive layer
      basic Decoder
      Decoder with global attention layer
      
      I also created some encoder and decoder with other RNN structure, these can be found out in 'models' filefolder'.
      
4. train.py is used to define how to run the structure. Two main functions are 'train' and 'trainiters'
    Please modify your dimensions and some other things if you changed the model structure such as layers, gru to lstm, unidirectional to bidirectional etc.
    
    5.run.py is used to run it!  outputs are loss function, some typical translation, attention viz and some others.



## CS 224N: Natural Language Processing with Deep Learning

# Compositional Pre-Training for Semantic Parsing with BERT

#### Arnaud Autef, Simon Hagege

#####Abstract

Semantic parsing - the conversion of natural language utterances to logical forms - is a typical natural language processing task. Its applications cover a wide variety of topics, such as question answering, machine translation, instruction following or regular expression generation. In our project, we investigate the performance of Transformer-based sequence-to-sequence models for semantic parsing. We compare a simple Transformer Encoder-Decoder model built on the work of (Vaswani et al., 2017) and a Encoder-Decoder model where the Encoder is BERT (Devlin et al., 2018), a Transformer with weights pre-trained to learn a bidirectional language model over a large dataset. Our architectures have been trained on semantic parsing data using data recombination techniques described in (Jia et al., 2016). We illustrated how Transformer-based Encoder Decoder architectures could be applied to semantic parsing. We evidenced the benefits of BERT as the Encoder of such architectures and pointed to some of the difficulties in fine-tuning it on limited data while the Decoder has to be trained from scratch. We have been mentored by Robin Jia from Stanford University's Computer Science Department.

#####Course

This project was part of the course CS 224N: Natural Language Processing with Deep Learning taught at Stanford University.
http://cs224n.stanford.edu
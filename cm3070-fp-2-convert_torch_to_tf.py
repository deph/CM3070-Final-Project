# -----------------------------------------------------------------------------------------
# 8. BERT Model Conversion: PyTorch to TensorFlow for Deployment
# -----------------------------------------------------------------------------------------

# Library: Import BERT model for sequence classification and tokenizer from Hugging Face
from transformers import TFBertForSequenceClassification, BertTokenizer

# Load pre-trained BERT model for sequence classification and the corresponding tokenizer
model_bert_tf = TFBertForSequenceClassification.from_pretrained("model_bert", from_pt=True)
tokenizer_bert_tf = BertTokenizer.from_pretrained("model_bert")

# Save the model and tokenizer to the specified directory
model_bert_tf.save_pretrained("saved_models/model_bert_tf")
tokenizer_bert_tf.save_pretrained("saved_models/model_bert_tf")

print("✅ TF model saved to saved_models")


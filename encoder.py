import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class RelationExtractor(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_relations=0, freeze_bert=False):
        """
        Initialize the relation extractor with a pre-trained BERT model.
        
        Args:
            model_name (str): The name of the pre-trained BERT model
            num_relations (int): Number of relation classes for classification
            freeze_bert (bool): Whether to freeze BERT parameters during training
        """
        super(RelationExtractor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.bert.config.hidden_size
        if num_relations > 0:
            self.classifier = nn.Linear(self.hidden_size, num_relations)
        else:
            self.classifier = None
        
        self.to(self.device)
        
    def create_template(self, sentence, head_entity, tail_entity):
        """
        Format the input into a cloze-style phrase as described in the paper.
        
        Args:
            sentence (str): The input sentence
            head_entity (str): The head entity in the sentence
            tail_entity (str): The tail entity in the sentence
            
        Returns:
            str: The formatted template with [MASK] token
        """
        v_tokens = ["[unused{}]".format(i) for i in range(1, 13)]
        
        # Create the template following T(x) = x [v0:n0-1] eh [vn0:n1-1] [MASK] [vn1:n2-1] et [vn2:n3-1]
        # Where n0=3, n1=6, n2=9, n3=12
        template = (
            f"{sentence} "
            f"{' '.join(v_tokens[0:3])} {head_entity} {' '.join(v_tokens[3:6])} [MASK] "
            f"{' '.join(v_tokens[6:9])} {tail_entity} {' '.join(v_tokens[9:12])}"
        )
        
        return template
    
    def get_latent_representation(self, sentence, head_entity, tail_entity):
        """
        Compute the latent representation z for the given sentence and entities.
        
        Args:
            sentence (str): The input sentence
            head_entity (str): The head entity in the sentence
            tail_entity (str): The tail entity in the sentence
            
        Returns:
            torch.Tensor: The latent representation z corresponding to [MASK] position
        """
        template = self.create_template(sentence, head_entity, tail_entity)
        
        inputs = self.tokenizer(template, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        mask_position = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1].item()
        
        outputs = self.bert(**inputs)
        
        z = outputs.last_hidden_state[0, mask_position, :]
        
        return z
    
    def forward(self, sentences, head_entities, tail_entities):
        """
        Forward pass through the model.
        
        Args:
            sentences (list): List of input sentences
            head_entities (list): List of head entities corresponding to each sentence
            tail_entities (list): List of tail entities corresponding to each sentence
            
        Returns:
            dict: Dictionary containing latent representations and optional relation logits
        """
        batch_size = len(sentences)
        latent_representations = []
        
        for i in range(batch_size):
            z = self.get_latent_representation(sentences[i], head_entities[i], tail_entities[i])
            latent_representations.append(z)
        
        latent_representations = torch.stack(latent_representations)
        
        if self.classifier is not None:
            relation_logits = self.classifier(latent_representations)
        else:
            relation_logits = None
        
        return {
            "latent_representations": latent_representations,
            "relation_logits": relation_logits
        }
 
    def predict_relation(self, sentence, head_entity, tail_entity):
        """
        Predict the relation between head and tail entities.
        
        Args:
            sentence (str): The input sentence
            head_entity (str): The head entity in the sentence
            tail_entity (str): The tail entity in the sentence
            
        Returns:
            dict: Prediction results containing relation ID, probabilities, and latent representation
        """
        if self.classifier is None:
            raise ValueError("This model doesn't have a classifier for relation prediction")
            
        z = self.get_latent_representation(sentence, head_entity, tail_entity)
        
        with torch.no_grad():
            logits = self.classifier(z)
            probs = torch.nn.functional.softmax(logits, dim=0)
        
        predicted_relation = torch.argmax(probs).item()
        
        return {
            "relation_id": predicted_relation,
            "probabilities": probs.cpu().detach().numpy(),
            "latent_representation": z.cpu().detach().numpy()
        }
    def get_similarity(self, z1, z2):
        """
        Compute cosine similarity between two latent representations.
        
        Args:
            z1, z2: Latent representations
            
        Returns:
            float: Cosine similarity score
        """
        z1_norm = torch.nn.functional.normalize(z1, p=2, dim=0)
        z2_norm = torch.nn.functional.normalize(z2, p=2, dim=0)
        similarity = torch.dot(z1_norm, z2_norm).item()
        
        return similarity

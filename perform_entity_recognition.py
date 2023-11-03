import spacy

def perform_entity_recognition(text):
    # Load the pre-trained model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Initialize empty list to store entities
    entities = []
    
    # Iterate over each entity in the document
    for entity in doc.ents:
        # Extract the entity text and label
        entity_text = entity.text
        entity_label = entity.label_
        
        # Append the entity and its label to the list
        entities.append((entity_text, entity_label))
    
    return entities

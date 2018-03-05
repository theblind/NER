# NER

NER (Named Entity Recognition) is a subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

To implement NER, we follow the following pipeline:
- Use bi-directional LSTM to process the input sequence.
- Use a fully connected layer to predict the category of named entity.
- Add CRF layer to calculate maximum likelihood of named entity for input sequence.

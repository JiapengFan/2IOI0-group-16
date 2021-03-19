def convertToOnehot(events_array, onehot_encoder):
    '''
    Returns one-hot encoded events.

    Args:
    events <class: 'np.array'>: Events to be converted.
    onehot_encoder <class: 'sklearn.preprocessing.LabelEncoder'>: Instance of OneHotEncoder that contains the defined mapping.

    Returns:
    label_encoder <class: 'sklearn.preprocessing.LabelEncoder'>: Mapping that will later be used to convert one-hot encoded events back to events in string.
    '''

    onehot_encoded = onehot_encoder.transform(events_array)

    return onehot_encoded

def convertToEvent(onehot_encoded: list, onehot_encoder):
    '''
    Returns events in string from one hot encoded events.

    Args:
    onehot_encoded <class: list>: One-hot encoded events.
    onehot_encoder <class: 'sklearn.preprocessing.LabelEncoder'>: Mapping used to convert one-hot encoded events back to events in string.

    Returns:
    events <class: list>: Events in string from one-hot encoded events.
    '''

    events = onehot_encoder.inverse_transform(onehot_encoded)

    return events
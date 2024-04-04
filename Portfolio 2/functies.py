# Importing standard libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa as lr
from datetime import datetime


def preditions_to_classes(y_pred, label_names):
    label_map = {i: class_name for i, class_name in enumerate(label_names)}
    predicted_classes = np.argmax(y_pred, axis=1)
    predictions = np.array([label_map[i] for i in predicted_classes])
    return predictions


def kaggle_csv(model, input_, suffix):
    """
    Deze functie maakt de voorspelling op de test dataset
    en vormt deze om tot een csv bestand om in te kunnen
    leveren op Kaggle.

    Parameters:
    ----------
    model : ML-model of DL-model
        De naam die is gegeven aan het ML-model of DL-model
        dat wordt gebruikt om te voorspellen

    input_ : pd.DataFrame of np.array
        De test data die in het model gaat

    suffix : str
        De 'tag' voor de naam van het csv
        bestand, zodat deze makkelijk te identificeren
        is na de submission.

    Returns:
    ----------
    None
        In plaats van een return maakt het een bestand
        aan in de map Kaggle Submissions.
    """
    # Aanmaken van de voorspelling
    y_pred = model.predict(input_)

    # Voorspelling omzetten van kans naar genre
    classes = preditions_to_classes(y_pred)

    # Aanmaken df met alleen filename en genre
    test_predictions_df = pd.DataFrame(
        {'filename': test_data['filename'],
        'genre': classes})

    # Aanmaken van tijd
    tijd = datetime.now().strftime("%m%d%H%M%S")

    # Aanmaken csv bestand met timestamp
    test_predictions_df.to_csv(
        f'Kaggle/voorspelling_{suffix}_{tijd}.csv',
        index=False)

    # Print voor conformatie
    print(f'voorspelling_{suffix}_{tijd}.csv has been saved!')

# Dataset label mappings for RESISC45 dataset

LABEL_MAPPING = {
    '0': 'airplane',
    '1': 'airport',
    '2': 'baseball_diamond',
    '3': 'basketball_court',
    '4': 'beach',
    '5': 'bridge',
    '6': 'chaparral',
    '7': 'church',
    '8': 'circular_farmland',
    '9': 'cloud',
    '10': 'commercial_area',
    '11': 'dense_residential',
    '12': 'desert',
    '13': 'forest',
    '14': 'freeway',
    '15': 'golf_course',
    '16': 'ground_track_field',
    '17': 'harbor',
    '18': 'industrial_area',
    '19': 'intersection',
    '20': 'island',
    '21': 'lake',
    '22': 'meadow',
    '23': 'medium_residential',
    '24': 'mobile_home_park',
    '25': 'mountain',
    '26': 'overpass',
    '27': 'palace',
    '28': 'parking_lot',
    '29': 'railway',
    '30': 'railway_station',
    '31': 'rectangular_farmland',
    '32': 'river',
    '33': 'roundabout',
    '34': 'runway',
    '35': 'sea_ice',
    '36': 'ship',
    '37': 'snowberg',
    '38': 'sparse_residential',
    '39': 'stadium',
    '40': 'storage_tank',
    '41': 'tennis_court',
    '42': 'terrace',
    '43': 'thermal_power_station',
    '44': 'wetland'
}

# Create inverse mapping (name to index)
IDX_TO_LABEL = {i: label for i, label in enumerate(LABEL_MAPPING.values())}
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_MAPPING.values())}

# Dataset information
DATASET_INFO = {
    'num_classes': 45,
    'size': {
        'train': 18900,
        'validation': 6300,
        'test': 6300
    },
    'sports_classes': [
        'baseball_diamond',
        'basketball_court',
        'golf_course',
        'ground_track_field',
        'tennis_court',
        'stadium'
    ]
}

import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


__locations_data = None
__model = None


def get_estimated_price(location, sqft, bath, bhk):
    global __model
    try:
        loc_index = __locations_data.index(location.lower())
    
    except IndexError:
        return "\nLocation not found in the dataset"
    except ValueError:
        return f"\n'{location}' not found in the dataset"

    
    X = np.zeros(4)
    ## so it works like we create a dummy array of zeros
    # in this case [0, 0, 0, 0]
    # now the values in the variable is updated to out inputs vaiables
    # i.e
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    
    #[sqft, bath, bhk, 0]
    X[3] = loc_index
    
    print(X)
    return (f"{__model.predict([X])[0]:.2f}")

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations_data
    global __model

    with open("./artifacts/locations.json",'r') as f:
        __locations_data = json.load(f)['locations_data']

    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")



def get_location_names():
    return __locations_data


if __name__ == '__main__':
    
    load_saved_artifacts()
    get_location_names()
    
    
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 3))
    print(get_estimated_price('Indira Nagar', 1000, 2, 3))
    print(get_estimated_price('Ejipura', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 7, 2 ))

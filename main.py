from src import *
import os

def main():
    choice = -1
    model = None
    os.system('cls')
    while choice != 7:
        print("===Digit Reckognision===")
        print("1. Load Model")
        print("2. Create Model")
        print("3. Train Model")
        print("4. Evaluate Model")
        print("5. Evaluate Image")
        print("6. Save Model")
        print("7. Exit")

        choice = int(input())
        os.system('cls')
        
        if choice == 1:
            model = load_model()
            print("Model Loaded")
        elif choice == 2:
            config = {
                "input_size": None,
                "hidden_size": None,
                "output_size": None,
                "num_layers": None,
                "activation": None
            }
            for key in config.keys():
                print("Enter:", key, end=' ')
                if key != "activation":
                    config[key] = int(input())
                else:
                    config[key] = input()
            model = create_model(config)
            print("Model Created")
        elif choice == 3:
            epoch = int(input("Enter Epoch:"))
            train(model, epoch)
        elif choice == 4:
            evaluate(model)
        elif choice == 5:
            path = input("Enter Path:")
            evaluate_image(model, path)
        elif choice == 6:
            save_model(model)
            print("Model Saved")

if __name__ == "__main__":
    main()
import torch
import os
import numpy as np
from PIL import Image
from model.model import ChessNet

def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.device = device
    return model

def predict_directory(model, test_dir, device):
    from PIL import Image
    correct = 0
    total = 0
    results = []
   
    for class_name in ['white', 'black', 'empty']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
           
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                true_label = class_name

                image = Image.open(img_path).convert('RGB')
                pred_class, probs = predictImage(model, image, device)

                is_correct = pred_class == true_label
                correct += is_correct
                total += 1
               
                results.append({
                    'image': img_path,
                    'true': true_label,
                    'predicted': pred_class,
                    'correct': is_correct,
                    'probabilities': {
                        cls: prob.item() 
                        for cls, prob in zip(['white', 'black', 'empty'], probs)
                    }
                })
   
    accuracy = 100 * correct / total if total > 0 else 0
    return results, accuracy

def predictImage(model, image, device):
    model.eval()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
        
    image_tensor = model.transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probs = torch.nn.functional.softmax(output, dim=1)
        return ['white', 'black', 'empty'][predicted.item()], probs[0]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('chessClassifier.pth', device)

    test_dir = "squaresTest"
    results, accuracy = predict_directory(model, test_dir, device)

    print(f"\nOverall Accuracy: {accuracy:.1f}%\n")
    print("Individual Results:")
    print("-" * 50)
   
    for result in results:
        print(f"\nImage: {os.path.basename(result['image'])}")
        print(f"True: {result['true']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Confidence: {max(result['probabilities'].values())*100:.1f}%")
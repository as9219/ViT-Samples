from torch import nn, optim, no_grad, argmax, uint8, cuda
import timeit # tracking training time
from tqdm import tqdm # progress bar

def train_model(model, train_dataloader, val_dataloader, device, ADAM_BETAS, LEARNING_RATE, ADAM_WEIGHT_DECAY, EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

    start = timeit.default_timer()
    for epoch in tqdm(range(EPOCHS), position=0, leave=True):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(uint8).to(device)
            y_pred = model(img)
            y_pred_label = argmax(y_pred, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())
            
            loss = criterion(y_pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with no_grad():
            for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_label["image"].float().to(device)
                label = img_label["label"].type(uint8).to(device)         
                y_pred = model(img)
                y_pred_label = argmax(y_pred, dim=1)
                
                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())
                
                loss = criterion(y_pred, label)
                val_running_loss += loss.item()
        val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Training Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Validity Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Training Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print(f"Validity Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
        print("-"*30)

    stop = timeit.default_timer()
    print(f"Training Time: {stop-start:.2f}s")

    cuda.empty_cache()
ChefBot is an AI program trained using Pytorch. The purpose of the program is to send recipes and basic ingredients when you ask it.

Interface: GUI - Tkinter

Database: intents.json file


Program summary: When the model enters the training stage, it will save a data.pth file. 
After that, the main.py file is connected to the Tkinter file named app.py. 
When you run the program, the GUI appears like a chat box message and you can start texting the ChefBot.
In the end when you want to close the execution, press ESCAPE to exit fullscreen and you can press X button on the window to close.

* After the execution, a file called chat_history.txt will be created. In that file you eill find ONLY the last execution's chat history, not all of them.

Requirements: Pytorch, NLTK, torchvision, numpy. The rest of the packages are built-in.
** I recommend to create a virtual environment for the program.

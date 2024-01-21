
# LLM_Chatbot
This LLM chat-bot is A Web App where users can upload their .pdf/.txt/.docx/.doc files having the texts and then they can ask questions from the chatbot related to that text and according to the question, it will answer based on the text provided.


## Run Locally

1) Clone the project

```bash
  git clone https://github.com/yashsehgal29/LLM_chatbot.git
```

2) Go to the project directory

```bash
  cd LLM_chatbot
```

3) Setup Virtual Environment

```bash
  pip install virtualenv
```
4) Create the Virtual Environment(let's call it venv)
```bash
  virtualenv venv
```

5) Activate the Virtual Environment

 5.1) Windows:
```bash
  .\Scripts\activate
```
 5.2) Linux:
```bash
  source venv/bin/activate
```
Your prompt will change to indicate that you are now operating within the virtual environment. It will look something like this <mark> (venv)user@host:~/venv$.</mark> 

6) Install the Requirements:
```bash
  pip install -r requirements.txt
``````
This will install all the libraries and dependencies

7) Create a file and save is as ".env"

8) In the .env file creat a variable named "REPLICATE_API_TOKEN" and Enter the token

```bash
  REPLICATE_API_TOKEN= "Enter your Replicate api token" 
``````
9) Finally Run the server using"
```bash
  streamlit run app.py
``````
## Authors

- [@Devongo](https://github.com/Dev-on-go)


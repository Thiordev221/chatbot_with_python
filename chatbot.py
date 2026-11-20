# Bibliothèques utilisées pour les variables d'environnement, l'API et le comptage des tokens.
import dotenv
import os
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer
import time

# Charge les variables définies dans le fichier .env.
dotenv.load_dotenv()


# Regroupe la configuration et les fonctionnalités principales du chatbot.
class Chatbot:
    def __init__(self):
        # Initialise le client compatible avec l'API OpenAI, configuré ici pour Groq.
        self.llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            )

        # Définit le message système qui donne son rôle et son style de réponse au bot.
        self.messages = [{
            'role' : 'system',
            'content' : """
            You are a helpful assistant. your name is Thior_Bot. Introduce yourself in the first message. give short and concise answers. But for histories, you can give very long text
            """,
        }]

        # Prépare les compteurs de tokens pour suivre l'utilisation de la conversation.
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

        self.iteration = 0

    def summarize_meesages(self):

        if self.iteration % 5 != 0:
            return

        system_messages = [message for message in self.messages if message['role'] == 'system']
        other_messages = [message for message in self.messages if message['role'] != 'system']

        early_messages, last_messages = (other_messages[:5], other_messages[5:])

        response = self.llm.chat.completions.create(
            model='openai/gpt-oss-20b',
            messages=[{
                'role' : 'system',
                'content' : '''
                    Summarize these  messages concisely. Keep key facts and decisions.
                '''
            },
            *early_messages,]
        )

        summarized_messages = [
            {
                'role' : 'system',
                'content' : f'''This is a summary of the five early messages : 
                {response.choices[0].message.content}'''
            },
        ]

        self.messages = [*system_messages, *summarized_messages, *last_messages]

        print('Summarization successful')

    def trim_messages(self, max_messages=4):
        if len(self.messages) < 4:
            return

        system_messages = [message for message in self.messages if message['role'] == 'system']
        other_messages = [message for message in self.messages if message['role'] != 'system']

        trimed_messages = other_messages[-max_messages:]

        self.messages = [*system_messages, *trimed_messages]


    def bot(self, message):
        # Ajoute le message de l'utilisateur à l'historique envoyé au modèle.
        self.messages.append({
            'role' : 'user',
            'content' : message,
        })

        # Demande une réponse au modèle en mode streaming pour l'afficher au fur et à mesure.
        response = self.llm.chat.completions.create(
            model='openai/gpt-oss-20b',
            messages=self.messages,
            temperature=0.7,
            stream=True
        )

        # Récupère les morceaux de réponse, les affiche et comptabilise les tokens fournis.
        response_text = ''
        for chunk in response:
            if chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end='', flush=True)
                    response_text += delta
                    time.sleep(0.05)

            if chunk.usage:
                self.total_tokens += chunk.usage.total_tokens
                self.input_tokens += chunk.usage.prompt_tokens
                self.output_tokens += chunk.usage.completion_tokens

        print()

        # Conserve la réponse dans l'historique pour donner le contexte aux prochains échanges.
        self.messages.append({
            'role' : 'assistant',
            'content' : response_text,
        })

        self.summarize_meesages()

        # Renvoie la réponse complète, en plus de l'avoir affichée dans le terminal.
        return response_text

    def chat(self):
        # Sélectionne l'encodeur servant à estimer le nombre de tokens du message utilisateur.
        encoding = tiktoken.get_encoding(encoding_name="cl100k_base")
        # Autre encodeur possible : tokenizer Hugging Face associé au modèle indiqué.
        # encoding = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=os.getenv("HF_TOKEN"))

        # Maintient la conversation interactive jusqu'à la demande de quitter ou à une limite.
        while True:

            self.iteration += 1

            # Lit le message, retire les espaces périphériques et normalise sa casse.
            user_input = input("Enter your message (type \"q\" to quit) : ").strip().lower()

            # Termine la boucle si l'utilisateur saisit la commande de sortie.
            if user_input == "q":
                print("Goodbye!")
                break

            # Mesure et affiche le nombre de tokens du message saisi.
            token_count = len(encoding.encode(user_input))
            print(f"Number of tokens in the message: {token_count}")

            # Refuse les messages dépassant la limite autorisée par le programme.
            if token_count > 10:
                print("Please enter a message that is less than 10 tokens.")
                continue

            # Arrête la conversation lorsque le total de tokens dépasse la limite globale.
            if self.total_tokens > 10_000:
                print("You have reached the maximum number of tokens.")
                break

            # Envoie le message au modèle et affiche sa réponse.
            response = self.bot(user_input)

    def report(self):
        # Affiche l'historique complet de la conversation, y compris le message système.
        for message in self.messages:
            print(message)

        # Affiche le total de tokens consommés et estime le coût selon les tarifs indiqués.
        print(f'sonsumed tokens : {self.total_tokens}')

        print(f'The cost : ${(self.input_tokens * 0.05 + self.output_tokens * 0.08)/1_000_000:.2f}')


# Point d'entrée qui crée le chatbot, lance la conversation puis affiche le bilan.
def main():
   bot = Chatbot()
   bot.chat()
   bot.report()


# Exécute main uniquement lorsque ce fichier est lancé directement.
if __name__ == "__main__":
    main()

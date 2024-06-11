import os
import click
import anthropic
from openai import OpenAI
from loguru import logger


@click.command()
@click.option(
    "--txt_path",
    type=click.Path(exists=True),
    prompt="Path to prompt TXT file",
    help="Path to the TXT file containing the prompt",
)
@click.option(
    "--tex_path",
    type=click.Path(exists=True),
    prompt="Path to appendix TEX file",
    help="Path to the TEX file to append as appendix",
)
@click.option(
    "--output_path",
    type=click.Path(),
    prompt="Output path for responses",
    help="Path to save the output responses",
)
def process_and_query(txt_path, tex_path, output_path):
    # Load the content from the TXT file
    with open(txt_path, "r", encoding="utf-8") as file:
        prompt_content = file.read()

    # Load the content from the TEX file
    with open(tex_path, "r", encoding="utf-8") as file:
        appendix_content = file.read()

    # Combine the contents
    combined_content = prompt_content + "\nAppendix (Report):\n" + appendix_content

    # Send to Anthropic using the SDK
    logger.info("Querying Anthropic...")
    anthropic_response = query_anthropic(combined_content)

    # Send to OpenAI using the SDK
    logger.info("Querying OpenAI...")
    openai_response = query_openai(combined_content)

    # Save the responses
    logger.info("Saving responses...")
    save_responses(anthropic_response, openai_response, output_path)


def query_anthropic(text):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=[{"role": "user", "content": text}],
    )
    return "\n".join([textblock.text for textblock in message.content])


def query_openai(text):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY="))
    chat_completion = client.chat.completions.create(
        model="gpt-4-turbo", messages=[{"role": "user", "content": text}]
    )
    return chat_completion.choices[0].message.content or ""


def save_responses(anthropic_response, openai_response, path):
    with open(path, "w", encoding="utf-8") as file:
        file.write("Claude 3 Opus\n")
        file.write(str(anthropic_response))
        file.write("\n\nOpenAI GPT4 Turbo\n")
        file.write(str(openai_response))


if __name__ == "__main__":
    process_and_query()

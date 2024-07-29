from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Type

from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam

from .models.trial_context import TrialContext
from .models.trial_loop import TrialLoop
from .utils import log


def run_evaluation(team_name: str, fn: Type[TrialLoop], num_trials=10,
                   characters: list[str] = None):
    """
    Run a trial for each character in the alphabet for a given team.
    :param team_name: team name
    :param fn: trial loop function
    :param num_trials: number of trials to run (default 10)
    :param characters: characters to run trials for (default all characters)
    :return: None
    """
    all_characters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                      "S", "T", "U", "V", "W", "X", "Y", "Z"]
    if characters is None:
        characters = all_characters

    for character in characters:
        if character not in all_characters:
            raise ValueError(f"Character {character} is not in the alphabet.")

    prohibited_team_name = ["logs"]
    if team_name in prohibited_team_name:
        raise ValueError(f"Team name {team_name} is prohibited.")

    team_path = Path(team_name)
    logging_path = team_path.parent / "logs"
    output_path = team_path / "raw"

    Path.mkdir(team_path, exist_ok=True)
    Path.mkdir(logging_path, exist_ok=True)
    Path.mkdir(output_path, exist_ok=True)

    log_file = logging_path / f"{team_name}_raw_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%Z%z')}.log"
    log_file.touch(exist_ok=True)

    for character in characters:
        log(log_file, f"Running trials for character {character} for team {team_name}")

        character_path = output_path / character
        Path.mkdir(character_path, exist_ok=True)

        for trial_number in range(1, num_trials + 1):
            log(log_file,
                f"Running trial {trial_number} for character {character} for team {team_name}")
            ctx = TrialContext(team_name, character, trial_number, log_file)
            if (ctx.get_output_file_path()).exists():
                log(log_file,
                    f"Trial {trial_number} for character {character} for team {team_name} already exists. Skipping.")
                continue

            __run_trial(ctx, fn)


def __run_trial(ctx: TrialContext, fn: Type[TrialLoop]):
    """
    Run a single trial.
    :param ctx: context containing trial information
    :param fn: trial loop function
    :return: None
    """
    log_file_path = ctx.get_log_file_path()

    try:
        final_response = fn.run(ctx, ctx.get_character())
    except (TimeoutError, ValueError) as e:
        log(log_file_path, f"Trial {ctx.get_trial_number()} failed with error: {e}")
        return

    with open(ctx.get_output_file_path(), "w", encoding="utf-8") as f:
        log(log_file_path, f"Trial {ctx.get_trial_number()} succeeded")
        f.write(final_response)


def chat_with_chatgpt(ctx: TrialContext,
                      messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam |
                                     ChatCompletionAssistantMessageParam], n=1) -> list[str]:
    """
    Chat with ChatGPT.
    :param ctx: context containing trial information
    :param messages: history of messages
    :param n: number of responses to generate (default 1)
    :return: response
    """
    model = "gpt-3.5-turbo-0125"
    temperature = 1
    seed = 42
    max_time = 120
    token_limit = 25000
    log_file_path = ctx.get_log_file_path()

    current_time = perf_counter()
    elapsed_time = current_time - ctx.get_start_time()
    log(log_file_path, f"Elapsed time: {elapsed_time}")

    if elapsed_time > max_time:
        log(log_file_path, f"Time limit exceeded. {elapsed_time} > {max_time}")
        raise TimeoutError(f"Time limit exceeded. {elapsed_time} > {max_time}")

    client = OpenAI(timeout=60.0)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        seed=seed,
        n=n,
    )

    responses = [chat_completion.choices[i].message.content for i in range(n)]
    ctx.add_prompt_token_count(chat_completion.usage.prompt_tokens)
    ctx.add_output_token_count(chat_completion.usage.completion_tokens)

    log(log_file_path, f"Messages: {messages}")
    log(log_file_path, f"Response: {responses}")
    log(log_file_path, f"Prompt token count: {chat_completion.usage.prompt_tokens}")
    log(log_file_path, f"Output token count: {chat_completion.usage.completion_tokens}")
    log(log_file_path, f"Total token count: {ctx.get_total_token_count()}")

    if ctx.get_total_token_count() > token_limit:
        log(log_file_path, f"Prompt token limit exceeded. {ctx.get_total_token_count()} > {token_limit}")
        raise ValueError(f"Prompt token limit exceeded. {ctx.get_total_token_count()} > {token_limit}")

    return responses

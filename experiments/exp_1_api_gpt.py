import os
import pandas as pd
from openpyxl import load_workbook
from openai import OpenAI
from dotenv import load_dotenv

import config
from .similarity import calculate_cosine_similarity

# init open ai client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_file_path():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(root_path, 'data/exp_1_api_gpt.xlsx')
    return file_path


def copy_qa_sheet_as_target(sheet_name):
    excel_path = get_file_path()
    wb = load_workbook(excel_path)

    # avoid name collision
    i = 1
    target_name = sheet_name
    while target_name in wb.sheetnames:
        target_name = f'{sheet_name}_{i}'
        i += 1

    # save sheet and return actual sheet name
    source_sheet = wb['QAs']
    new_sheet = wb.copy_worksheet(source_sheet)
    new_sheet.title = target_name
    wb.save(excel_path)
    return target_name


def ask_open_ai(q):
    res = client.responses.create(
        model=config.OPEN_AI_MODEL,
        instructions='',
        input=q,
    )
    return res.output_text


def asking_each_questions(sheet_name):
    excel_path = get_file_path()

    # read and deal data
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    questions_col = df['three_asking_ways']
    answers = []
    for i, q in enumerate(questions_col):
        answer = ask_open_ai(q)
        answers.append(answer)
    df['answers'] = answers

    # similarity
    df['standard_answers'] = df['standard_answers'].ffill()
    similarity_list = calculate_cosine_similarity(
        df['standard_answers'].astype(str).tolist(),
        df['answers'].astype(str).tolist(),
    )
    df['similarity'] = similarity_list

    # read file and column by openpyxl
    wb = load_workbook(excel_path)
    ws = wb[sheet_name]
    header = [cell.value for cell in ws[1]]
    answer_column_index = header.index('answers') + 1
    similarity_column_index = header.index('similarity') + 1

    # write answers row by row
    for i, v in enumerate(df['answers'], start=2):
        ws.cell(row=i, column=answer_column_index, value=v)

    # write similarity row by row
    for i, v in enumerate(df['similarity'], start=2):
        ws.cell(row=i, column=similarity_column_index, value=v)

    # use this to save for keeping style of the sheet
    wb.save(excel_path)


if __name__ == '__main__':
    target_sheet = copy_qa_sheet_as_target('exp_1_API_GPT')
    asking_each_questions(target_sheet)




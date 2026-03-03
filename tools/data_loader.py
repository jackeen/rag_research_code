from pathlib import Path
from enum import Enum


class BookNames(Enum):
    INTRO_WEB_DEV_1 = "1.pdf"
    RESPONSIVE_WEB_DESIGN_2 = "2.pdf"
    LEARNING_REACT_4 = "4.pdf"
    DESIGN_PATTERN_5 = "5.pdf"
    STRUCTURE_INTERPRETATION_6 = "6.pdf"
    SOCIAL_MARKETING_8 = "8.pdf"


class References(Enum):
    INTRO_WEB_DEV_1 = "Mendez, M. (2014). The Missing Link: An Introduction to Web Development and Programming. Open SUNY Textbooks."
    RESPONSIVE_WEB_DESIGN_2 = "APA"
    LEARNING_REACT_4 = "Banks, A., & Porcello, E. (2017). Learning React: functional web development with React and Redux. ' O'Reilly Media, Inc.'."
    DESIGN_PATTERN_5 = "Freeman, E., & Robson, E. (2020). Head first design patterns. O'Reilly Media."
    STRUCTURE_INTERPRETATION_6 = "Abelson, H., & Sussman, G. J. (1996). Structure and interpretation of computer programs (p. 688). The MIT Press."
    SOCIAL_MARKETING_8 = "Donovan, R., & Henley, N. (2010). Principles and practice of social marketing: an international perspective. Cambridge University Press."


def get_book_path(book_file_name: str) -> Path:
    """
    Get the experiment original book path as the data.
    :param book_file_name:
    :return: the absolute path of the book
    """
    return Path(__file__).parent.parent.absolute().joinpath(f"books/{book_file_name}")


def get_excel_data_path(excel_name: str) -> Path:
    """
    Get the experiment result Excel data path by given file name.
    :param excel_name: the Excel file name
    :return: the absolute path of the Excel file
    """
    return Path(__file__).parent.parent.absolute().joinpath(f"data/{excel_name}.xlsx")


def get_csv_data_path(csv_name: str) -> Path:
    return Path(__file__).parent.parent.absolute().joinpath(f"data/csv/{csv_name}.csv")


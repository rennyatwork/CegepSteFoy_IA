#!/usr/bin/python
## beautifulSoup example (not scrapy)
import requests

from os import system
from sys import exit
from time import sleep
from requests.exceptions import ConnectionError

from bs4 import BeautifulSoup

from article import Article


URL = "http://www.geeksforgeeks.org/"
articles = []

CHOICE_TO_CATEGORY_MAPPING = {
    1: "c",
    2: "c-plus-plus",
    3: "java",
    4: "python",
    5: "fundamentals-of-algorithms",
    6: "data-structures",
}


def display_menu():
    print("Choose category to scrape: ")
    print("1. C Language")
    print("2. C++ Language")
    print("3. Java")
    print("4. Python")
    print("5. Algorithms")
    print("6. Data Structures")



def get_category_choice():
    choice = int(input("Enter choice: "))
    try:
        category_url = CHOICE_TO_CATEGORY_MAPPING[choice]
    except KeyError:
        print("Wrong Choice Entered. Exiting!")
        exit(1)
    return category_url


def save_articles_as_html_and_pdf():
    print("All links scraped, extracting articles")
    # Formatage du code HTML pour les articles
    all_articles = (
        "<!DOCTYPE html>"
        "<html><head>"
        '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />'
        '<link rel="stylesheet" href="style.min.css" type="text/css" media="all" />'
        '<script src="https://cdn.rawgit.com/google/code-prettify/master/loader/run_prettify.js"></script>'
        "</head><body>"
    )

    all_articles += (
        '<h1 style="text-align:center;font-size:40px">'
        + category_url.title()
        + " Archive</h1><hr>"
    )
    all_articles += '<h1 style="padding-left:5%;font-size:200%;">Index</h1><br/>'

    for x in range(len(articles)):
        all_articles += (
            '<a href ="#'
            + str(x + 1)
            + '">'
            + '<h1 style="padding-left:5%;font-size:20px;">'
            + str(x + 1)
            + ".\t\t"
            + articles[x].title
            + "</h1></a> <br/>"
        )
    for x in range(len(articles)):
        all_articles += (
            '<hr id="' + str(x + 1) + '">' + articles[x].content.decode("utf-8")
        )

    all_articles += """</body></html>"""
    html_file_name = "Result" + category_url.title() + ".html"
    html_file = open(html_file_name, "wb")
    html_file.write(all_articles.encode("utf-8"))
    html_file.close()

    #pdf_file_name = "Result" + category_url.title() + ".pdf"
    #print("Generating PDF " + pdf_file_name)
    #html_to_pdf_command = "wkhtmltopdf " + html_file_name + " " + pdf_file_name
    #system(html_to_pdf_command)


def scrape_category(category_url):
    try:
        soup = BeautifulSoup(requests.get(URL + category_url).text)
    except ConnectionError:
        print("Impossible de se connecter à Internet! Veuillez verifier votre connexion et réessayer.")
        exit(1)

    # Sélection des liens qui se trouvent dans la page de catégorie
    links = [a.attrs.get("href") for a in soup.select("article li a")]
    # Suppression des liens pour les catégories avec ancre sur la même page
    links = [link for link in links if not link.startswith("#")]

    print("Trouvé: " + str(len(links)) + " liens")
    i = 1

    # Parcourez chaque lien pour trouver un article et l'enregistrer.
    for link in links:
        try:
            if i % 10 == 0:
                sleep(5)  # Sleep for 5 seconds before scraping every 10th link
            link = link.strip()
            print("Scraping link no: " + str(i) + " Link: " + link)
            i += 1
            link_soup = BeautifulSoup(requests.get(link).text)
            # Remove the space occupied by Google Ads (Drop script & ins node)
            [script.extract() for script in link_soup(["script", "ins"])]
            for code_tag in link_soup.find_all("pre"):
                code_tag["class"] = code_tag.get("class", []) + ["prettyprint"]
            article = link_soup.find("article")
            # Maintenant, ajoutez cet article à la liste de tous les articles
            page = Article(
                title=link_soup.title.string, content=article.encode("UTF-8")
            )
            articles.append(page)
        # Parfois suspendu. Alors Ctrl ^ C, et essayez le lien suivant.
        # Découvrez la raison et améliorez cela.
        except KeyboardInterrupt:
            continue
        except ConnectionError:
            print("Internet déconnecté! Veuillez vérifier votre connexion et réessayer.")
            if articles:
                print("Création de PDF de liens grattés jusqu'à présent.")
                break
            else:
                exit(1)


if __name__ == "__main__":
    display_menu()
    category_url = get_category_choice()
    scrape_category(category_url)
    save_articles_as_html_and_pdf()
import requests
from bs4 import BeautifulSoup


def download(uid, path):
    url = f"https://www.facebook.com/profile.php?id={uid}"

    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    meta_tag = soup.find('meta', property='og:image')
    if meta_tag and 'content' in meta_tag.attrs:
        r = requests.get(meta_tag['content'])
        with open(path, 'wb') as file:
            file.write(r.content)
        return True
    else:
        return None


download("100009056020853",'image.jpg')

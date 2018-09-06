#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-06 10:24:03 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-06 10:24:03 
 * @Desc: a simple shell for downloading all ECCV 2018 papers.
            This is referred from CVer(Wechat public account)
            
        There are all papers in Baidu Netdisk which are download by this shell :
'''

from multiprocessing import Pool
import requests
from bs4 import BeautifulSoup
import traceback
import re
import os

prefix = "http://openaccess.thecvf.com/"
directory = "papers_ECCV2018/"


def get_pdf(data):
    href, title = data
    name = re.sub(r'[\\/:*?"<>|]', "", title)
    if os.path.isfile(directory+name):
        print("This file already exists, skip {}".format(name))
        return
    try:
        print(prefix+href)
        content = requests.get(prefix+href).content()
        with open(directory+"{}.pdf".format(name), "wb") as f:
            f.write(content)
        print("Finish downloading {}".format(title))
    except:
        print('Error when downloading {}'.format(name))
        # print(traceback.format_exc())


if __name__=="__main__":
    if not os.path.exists(directory):
        os.mkdir(directory)

    html = requests.get(prefix+"ECCV2018.py").content
    soup = BeautifulSoup(html, "lxml")
    a_list = soup.findAll("a")
    title_list = soup.findAll("dt", {"class": 'ptitle'})
    title_list = [_.text for _ in title_list]
    pdf_list = []
    for item in a_list:
        if item.text.strip() == "pdf":
            href = item.get("href").strip()
            pdf_list.append(href)
    assert len(pdf_list) == len(title_list), "numbers of title and pdf not equal"
    print("Find {} papers".format(len(pdf_list)))

    pool = Pool(10)
    pool.map(get_pdf, zip(pdf_list, title_list))

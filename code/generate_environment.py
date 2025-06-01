import os
import streamlit as st
import langchain
import langchain_huggingface
import langchain_community
import chromadb
import torch
import sentence_transformers
import pypandoc
import fitz
import fpdf
import requests
import bs4
import sklearn
import numpy

print("Please type 'pipreqs' in the terminal within the code folder to generate the requirements.txt file.")
print("After it is done, you may move the requirements.txt file outside of the code folder so that it may be in the root folder of this GitHub repo.")
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

def document_to_table(document):
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    trace0 = go.Table(
      header = dict(
        values = [['<b>EXPENSES</b>'],
                      ['<b>Q1</b>'],
                      ['<b>Q2</b>'],
                      ['<b>Q3</b>'],
                      ['<b>Q4</b>']],
        line = dict(color = '#506784'),
        fill = dict(color = headerColor),
        align = ['left','center'],
        font = dict(color = 'white', size = 12)
      ),
      cells = dict(
        values = [
          [['Salaries', 'Office', 'Merchandise', 'Legal', '<b>TOTAL</b>']],
          [[1200000, 20000, 80000, 2000, 12120000]],
          [[1300000, 20000, 70000, 2000, 130902000]],
          [[1300000, 20000, 120000, 2000, 131222000]],
          [[1400000, 20000, 90000, 2000, 14102000]]],
        line = dict(color = '#506784'),
        fill = dict(color = [rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]),
        align = ['left', 'center'],
        font = dict(color = '#506784', size = 11)
        ))

    data=[trace0]
    fig=go.Figure(data=data)    
    
    return fig


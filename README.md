# H**ome task 3** intern data scientist, Aramco Inn.

[Link](https://www.notion.so/Intern-data-scientist-Aramco-Inn-26a38b74254042428e42dbb5bc28d6b8?pvs=21) to the job description.

## Context

Before diving into the task, let's briefly go over some key concepts of our work. In oil&gas research we deploy a ML apps using standalone applications.

## Task

Your task is to pack [Streamlit](https://streamlit.io) ML application into working executable `.msi.` file.

## Requirements

You are expected to address the following points in your solution:

1. Get familiar with [ETNA](https://github.com/tinkoff-ai/etna) time-series library and concept of time-series back-testing.
2. Use [CatBoostPerSegment](https://etna-docs.netlify.app/api/etna.models.catboost.catboostpersegmentmodel#etna.models.catboost.CatBoostPerSegmentModel) and [pipeline](https://etna-docs.netlify.app/api/etna.pipeline.pipeline.pipeline#etna.pipeline.pipeline.Pipeline) to build and validate your forecasting model. You basically may use code from [Get Started](https://github.com/tinkoff-ai/etna#get-started) of the library. 
3. Build very simple Streamlit offline web-app where user can train and validate model. Make user able to choose ETNA transforms of his choice (just a few of them) via Streamlit API.
4. Visualize the results of model backtest and forecasts in the app.
5. Pack it into excecutable`.msi` installer using [Briefcase](https://briefcase.readthedocs.io/en/latest/). Install the app on your machine and test it. It must work.
6. The executable offline application is required to utilize the CatBoostPerSegment and the pipeline from the ETNA library and to visualize forecasts. This `.msi` installer should open a window in a web browser where the CatBoost model within the pipeline is executed. Connection to the internet is not allowed.
7. Provide a concise `ReadMe.txt` with instructions how to build the standalone app. **The instruction should have the steps required to go through in order to build the `.msi` that can be executed offline by final user.** 
8. Send the zip archive with `*.py` file(-s), config, `requirements.txt`, `ReadMe.txt` and any other required files to [@voskresenskiianton](http://voskresenskiianton.t.me/). Keep the size of the archive under 5 MB.

I hope this task provides you with a challenging and exciting opportunity to learn and apply your skills. Good luck!

## Final remarks:

- You are free to use a dataset of your choice.
- The `.msi` app installer must work. It really does.
- Do not publish the app to the web. That is not the aim of the home task.
- Make sure that you code and steps in `ReadMe.txt` are reproducible in a new Python environment.
- Do not use any other package library except Briefcase. We use this one for our projects and will continue to use it.

## F. A. Q.

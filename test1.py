from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse, PlainTextResponse
from starlette.templating import Jinja2Templates
from fastai.vision import *
import uvicorn
from datauri import DataURI

from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp


templates = Jinja2Templates(directory='templates')

app = Starlette(debug=True)
app.mount('/static', StaticFiles(directory='statics'), name='static')



fruit_images_path = Path("../../data/fruit")
fruit_learner = load_learner(fruit_images_path, 'export_v3.pkl')
print(fruit_learner.data.classes)



@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


#Creat SQL db of fruit/veg and associated GHG data rather than writing strings like a pleb
def get_ghg_data(class_pred):
   apple_string = "An apple a day for a year is equivalent to  driving a regular petrol car 32 miles or heating an average British home for 2 days"
   if(class_pred == "apples"):
    return apple_string
   else:
        return "balls"
   

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = fruit_learner.predict(img)
    print(outputs)
    class_string = pred_class.obj
    ghg_info = get_ghg_data(class_string)
    #return PlainTextResponse('Class: '+ class_string + ". \n" + ghg_info)

    return JSONResponse({ 'classification': class_string, 'ghg': ghg_info })

@app.route("/")
def form(request):
    template = "homepage_v2.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context)
    
    
@app.route('/webcam')
async def homepage(request):
    template = "webcam.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context)


@app.route('/return_image', methods=['POST'])
async def return_image(request):
        res = await request.body()
        res = res.decode("utf-8")
        uri = DataURI(res)
        print(uri.data)
        return predict_image_from_bytes(uri.data)
    
        

@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


@app.route('/error')
async def error(request):
    """
    An example error. Switch the `debug` setting to see either tracebacks or 500 pages.
    """
    raise RuntimeError("Oh no")


@app.exception_handler(404)
async def not_found(request, exc):
    """
    Return an HTTP 404 page.
    """
    template = "404.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    """
    Return an HTTP 500 page.
    """
    template = "500.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8009)
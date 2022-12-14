var today = new Date();
var year = today.getFullYear()
var month = (today.getMonth()) + 1
var day = today.getDate()
console.log(year)
console.log(month)
console.log(day)

fetch("output/statistics.json")
    .then(Response => Response.json())
    .then(data => {
        spam = data[year][month][day]
        document.getElementById('spam').innerText = "Spam Emails: " + spam['Spam']
        document.getElementById('ham').innerText = "Non Threatening Emails: " + spam['Ham']  
    })
document.addEventListener("DOMContentLoaded", function (event) {

    let intervalId = setInterval(() => {
        var div = document.getElementsByClassName("description")[0]

        if (div) {
            clearInterval(intervalId);
            updateSwagger(div);
        }
    }, 200);

});

function updateSwagger(div) {
    fetch("/content/swagger/description.html").then(response => {
        response.text().then(txt => {
            div.innerHTML += txt;     
        })
    })

    // let $jwt = $("#operations-Account-post_account_authenticate .responses-table tr.response pre code span").eq(4).html().replace("\"", "");
    var link = document.querySelector("link[rel*='icon']") || document.createElement('link');;
    document.head.removeChild(link);
    link = document.querySelector("link[rel*='icon']") || document.createElement('link');
    document.head.removeChild(link);
    link = document.createElement('link');
    link.type = 'image/x-icon';
    link.rel = 'shortcut icon';
    link.href = '/content/swagger/favicon.ico';
    document.getElementsByTagName('head')[0].appendChild(link);
    window.localStorage.removeItem('access-token');
    /*
    var html = document.getElementById('swagger-ui').innerHTML;
    html = html.replace(/localhost/g, "api.reclaim.ai");
    document.getElementById('swagger-ui').innerHTML = html;
    */
}

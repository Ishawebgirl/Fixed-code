export abstract class ApiBase {
  protected transformOptions(options: RequestInit) {
    options.credentials = 'include'

    return Promise.resolve(options)
  }
}

/*
var overlayTimer: ReturnType<typeof setTimeout>
const apiRootUrl = process.env.REACT_APP_API_URL

const showOverlay = (isMakingRequest: boolean) => {
  const overlay = document.getElementById('overlay')

  if (!overlay) return

  if (isMakingRequest) {
    overlayTimer = setTimeout(() => {
      overlay.classList.add('enabled')
    }, 500)
  } else {
    clearTimeout(overlayTimer)
    overlay.classList.remove('enabled')
  }
}
*/
/*
  axiosRequest.interceptors.request.use((config) => {
    let isShowOverlay = true;
    
    if (config.url?.endsWith("document")) {
      config.headers[contentTypeHeaderKey] = "multipart/form-data"; 
    } else {
      config.headers[contentTypeHeaderKey] = "application/json";
    } 
  
    config.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups";
  
    if ((config.url == authEndPoint || config.url == reauthEndPoint) && config?.data?.refreshToken !== undefined)
      isShowOverlay = false;
     
    showOverlay(isShowOverlay); 
    return config;
  }, (error) => {
    showOverlay(false);
    return Promise.reject(error);
  });
  
  axiosRequest.interceptors.response.use((config) => {
    // config.headers["Access-Control-Allow-Origin"] = "*";
    
    showOverlay(false);
    return config;
  }, (error) => {
    showOverlay(false);
    return Promise.reject(error);
  });
  */

function titleCase(str) {
   var splitStr = str.toLowerCase().split(' ');
   for (var i = 0; i < splitStr.length; i++) {
       // You do not need to check if i is larger than splitStr length, as your for does that for you
       // Assign it back to the array
       splitStr[i] = splitStr[i].charAt(0).toUpperCase() + splitStr[i].substring(1);     
   }
   // Directly return the joined string
   
   return splitStr.join(' ');
}

function strDate(dateValue) {
	var output = "";
	output += dateValue.getDate() + " de ";
	
	var locale = "es-mx",
      month = dateValue.toLocaleString(locale, { month: "long" });
    output += month + " del ";

	output += dateValue.getFullYear();
	
	return output;
}

function toggle_visibility(id, refCaller) {
    var e = document.getElementById(id);
    if(e.style.display == 'block'){
      e.style.display = 'none';
      refCaller.innerHTML = "<b>Mostrar</b>";
      refCaller.title = "Mostar";
    }
    else{
      e.style.display = 'block';
      refCaller.innerHTML = "<b>Ocultar</b>";
      refCaller.title = "Ocultar";
    }
    }

// RESPONSIVE CANVAS START
function dynamicCanvas(id){
    var canvas = document.getElementById(id);
    console.log(canvas.width, canvas.height);
    canvas.width = window.innerWidth;
    var heightRatio = 1.5;
    canvas.height = canvas.width * heightRatio;
}
// RESPONSIVE CANVAS END
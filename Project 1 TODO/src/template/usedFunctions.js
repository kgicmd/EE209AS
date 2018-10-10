function markAsDone(event){
  if (window.confirm("Mark as done: ")){
    var eventName{
      "eventName" = event
    }

    $.ajax({
        url: 'mrkDone',
        data: eventName,
        dataType : json,
        success: function () {
            if (data["status"] == "Success!") {
              alert("Success!");
            }
            else {
              alert("Fail to connect with DataBase, contact admin!");
            }
        }
      })
  }
};

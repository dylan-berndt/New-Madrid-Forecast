import {useEffect, useState, useRef} from 'react';
import React from 'react';
import './style.css';

export function formatDate(date, addTime=true) {
    const d = new Date(date * 1000);
    var minutes = d.getMinutes().toString();
    if (minutes.length == 1) {
        minutes = "0" + minutes;
    }
    var hours = d.getHours();
    const meridian = hours > 12 ? "AM" : "PM"
    hours = hours % 12;
    hours = hours == 0 ? 12 : hours;
    const time = " " + hours + ":" + minutes + " " + meridian;
    return (d.getMonth() + 1) + "/" + d.getDate() + (addTime ? time : "");
}

function RangeHandle({position, onMouseDown, left}) {
    const style = {
        left: position
    }
    return <div className="RangeHandle" style={style} onMouseDown={() => onMouseDown(left)}/>
}

export function RangeSlider({setLeft, setRight, maxLeft, maxRight}) {
    const [leftPosition, setLeftPosition] = useState(-2);
    const [rightPosition, setRightPosition] = useState(0);

    const sliderRef = useRef(null);
    const draggingRef = useRef(false);

    const draggingLeftRef = useRef(true);

    const [width, setWidth] = useState(0);

    useEffect(() => {
        setWidth(sliderRef.current.offsetWidth - 2);
        setRightPosition(sliderRef.current.offsetWidth - 27);
    }, [sliderRef.current]);

    const onMouseMove = (event) => {
        if (!draggingRef.current) return;
        const slider = sliderRef.current;
        if (!slider) return;

        var bounds = slider.getBoundingClientRect();
        var boxX = event.clientX - bounds.left;
        boxX = Math.max(-2, Math.min(width - 25, boxX - 20));

        // console.log(draggingLeft);

        if (draggingLeftRef.current) {
            boxX = Math.min(rightPosition - 25, boxX);

            var x = boxX / (bounds.right - bounds.left);
            x = Math.max(0, Math.min(1, x));
            var boundsLeft = maxLeft + (maxRight - maxLeft) * x;

            setLeft(boundsLeft);
            setLeftPosition(boxX);
        }
        else {
            boxX = Math.max(leftPosition + 25, boxX);

            var x = boxX / (bounds.right - bounds.left);
            x = Math.max(0, Math.min(1, x));
            var boundsLeft = maxLeft + (maxRight - maxLeft) * x;

            setRight(boundsLeft);
            setRightPosition(boxX);
        }
    }

    const onMouseDown = (left) => {
        draggingRef.current = true;
        draggingLeftRef.current = left;
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);
    };

    const onMouseUp = () => {
        draggingRef.current = false;
        // setDraggingLeft(null);
        // console.log(null);
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
    };

    return <div className="RangeSlider" ref={sliderRef}>
        <RangeHandle position={leftPosition} onMouseDown={onMouseDown} left={true}/>
        <RangeHandle position={rightPosition} onMouseDown={onMouseDown} left={false}/>
    </div>
}

export const CustomTooltip = ({active, payload, label}) => {
    if (active && payload && payload.length) {
        return (
            <div className="CustomTooltip">
                <p className="label">{`${formatDate(label)}`}</p>
                {
                    payload.map((element) => {
                        return <p color={element.color}>{`${element.name}: ${element.value} ft`}</p>
                    })
                }
            </div>
        )
    }
}
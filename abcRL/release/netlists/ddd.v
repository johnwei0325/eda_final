module top_1598227639_809568180_776209382_1234615 (a, b, c, d, e, f, g, h, o);
 input a, b, c, d, e, f, g, h;
 output o;
 wire k;
 and g1(c,y1,y2);
 and g2(d,y2,y3);
 and g3(e,y3,y4);
 and g4(f,y4,y5);
 and g5(g,y5,y6);
 nand g7(h, y6,o);
 and g0(a,b,y1);
endmodule